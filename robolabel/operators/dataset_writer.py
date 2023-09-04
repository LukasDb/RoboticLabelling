import asyncio
import json
import open3d as o3d
import numpy as np
import signal
import logging
from pathlib import Path
from PIL import Image
from dataclasses import dataclass


from ..labelled_object import LabelledObject
from ..camera import Camera
from robolabel.lib.geometry import invert_homogeneous
from robolabel.lib.exr import write_exr


class DelayedKeyboardInterrupt:
    def __init__(self, index) -> None:
        self.index = index

    def __enter__(self) -> None:
        self.signal_received = False
        self.old_handler: signal._HANDLER = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame) -> None:
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback) -> None:
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)  # type: ignore


@dataclass
class WriterSettings:
    use_writer: bool = True
    output_dir: str = "dataset"
    _is_pre_acquisition: bool = False


class DatasetWriter:
    def __init__(self) -> None:
        self._active_objects: list[LabelledObject] = []
        self.is_pre_acquisition: bool = True

    def setup(self, objects: list[LabelledObject], settings: WriterSettings) -> None:
        self._active_objects = objects
        self._output_dir = Path(settings.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, cam: Camera) -> None:
        """capture datapoint"""
        cam_dir = self._output_dir / cam.name
        self._data_dir = cam_dir / "gt"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._rgb_dir = cam_dir / "rgb"
        self._rgb_dir.mkdir(parents=True, exist_ok=True)
        self._mask_dir = cam_dir / "mask"
        self._mask_dir.mkdir(parents=True, exist_ok=True)
        self._depth_dir = cam_dir / "depth"
        self._depth_dir.mkdir(parents=True, exist_ok=True)

        existing_images = list(self._rgb_dir.glob("*.png"))
        existing_indices = [int(p.stem.split("_")[1]) for p in existing_images]
        if len(existing_indices) == 0:
            dataset_index = 0
        else:
            dataset_index = max(existing_indices) + 1

        # handle keyboard interrupt or exceptions
        with DelayedKeyboardInterrupt(dataset_index):
            try:
                self._generate_data(cam, dataset_index)
            except Exception as e:
                self._cleanup(dataset_index)
                raise e

    def _generate_data(self, cam: Camera, dataset_index: int):
        logging.debug(f"Generating data #{dataset_index}")
        frame = cam.get_frame()

        assert frame.rgb is not None, "RGB frame is None"
        assert frame.depth is not None, "Depth frame is None"

        rgb = Image.fromarray(frame.rgb)
        rgb.save(self._rgb_dir / f"rgb_{dataset_index:04}.png")

        write_exr(
            self._depth_dir / f"depth_{dataset_index:04}.exr",
            {"R": frame.depth.astype(np.float16)},
        )

        masks: dict[str, np.ndarray] = {}
        visib_mask = np.zeros((cam.height, cam.width), dtype=np.uint8)
        for object_id, obj in enumerate(self._active_objects, start=1):
            unoccluded_mask = self._render_object_mask(obj, cam)
            masks[f"{object_id:04}.R"] = unoccluded_mask.astype(np.float16)
            occluded_mask = self._calculate_occluded_mask(unoccluded_mask, obj, frame.depth)
            visib_mask[occluded_mask == 1] = object_id

        masks["visib.R"] = visib_mask.astype(np.float16)

        write_exr(self._mask_dir / f"mask_{dataset_index:04}.exr", masks)

        obj_list = []
        for object_id, obj in enumerate(self._active_objects, start=1):
            px_count_visib = np.count_nonzero(visib_mask == object_id)
            bbox_visib = self._get_bbox(visib_mask, object_id)

            obj_mask = masks[f"{object_id:04}.R"]
            bbox_obj = self._get_bbox(obj_mask, 1)
            px_count_all = np.count_nonzero(obj_mask == 1)
            px_count_valid = np.count_nonzero(frame.depth[visib_mask == object_id])
            visib_fract = 0.0 if px_count_all == 0 else px_count_visib / px_count_all

            obj_list.append(
                {
                    "class": obj.name.split(".")[0],
                    "object id": object_id,
                    "pos": list(obj.get_position()),
                    "rotation": list(obj.get_orientation().as_quat(canonical=True)),
                    "bbox_visib": bbox_visib,
                    "bbox_obj": bbox_obj,
                    "px_count_visib": px_count_visib,
                    "px_count_valid": px_count_valid,
                    "px_count_all": px_count_all,
                    "visib_fract": visib_fract,
                }
            )

        cam_pos = cam.get_position()
        cam_rot = cam.get_orientation()
        cam_matrix = cam.intrinsic_matrix

        meta_dict = {
            "cam_rotation": list(cam_rot.as_quat(canonical=True)),
            "cam_location": list(cam_pos),
            "cam_matrix": np.array(cam_matrix).tolist(),
            "objs": list(obj_list),
        }

        with (self._data_dir / f"gt_{dataset_index:05}.json").open("w") as F:
            json.dump(meta_dict, F, indent=2)

    def _get_bbox(self, mask, object_id):
        y, x = np.where(mask == object_id)
        if len(y) == 0:
            return [0, 0, 0, 0]
        x1 = np.min(x).tolist()
        x2 = np.max(x).tolist()
        y1 = np.min(y).tolist()
        y2 = np.max(y).tolist()
        return [x1, y1, x2, y2]

    def _cleanup(self, dataset_index):
        gt_path = self._data_dir / f"gt_{dataset_index:05}.json"
        if gt_path.exists():
            logging.debug(f"Removing {gt_path}")
            gt_path.unlink()

        rgb_path = self._rgb_dir / f"rgb_{dataset_index:04}.png"
        if rgb_path.exists():
            logging.debug(f"Removing {rgb_path}")
            rgb_path.unlink()

        mask_path = self._mask_dir / f"mask_{dataset_index:04}.exr"
        if mask_path.exists():
            logging.debug(f"Removing {mask_path}")
            mask_path.unlink()

        depth_path = self._depth_dir / f"depth_{dataset_index:04}.exr"
        if depth_path.exists():
            logging.debug(f"Removing {depth_path}")
            depth_path.unlink()

    def _render_object_mask(self, obj: LabelledObject, cam: Camera) -> np.ndarray:
        """render object mask from camera"""
        scene = o3d.t.geometry.RaycastingScene()

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh)
        mesh_t.transform(obj.pose)
        mesh_id = scene.add_triangles(mesh_t)

        # frustum = o3d.geometry.LineSet.create_camera_visualization(
        #     cam.width,
        #     cam.height,
        #     cam.intrinsic_matrix,
        #     invert_homogeneous(cam.pose),
        #     1.0,
        # )

        # mesh = o3d.geometry.TriangleMesh(obj.mesh)
        # mesh.compute_vertex_normals()
        # mesh.paint_uniform_color(np.array(obj.semantic_color) / 255.0)
        # mesh.transform(obj.pose)
        # o3d.visualization.draw_geometries([mesh, frustum])  # type: ignore

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(cam.intrinsic_matrix),
            o3d.core.Tensor(invert_homogeneous(cam.pose)),
            cam.width,
            cam.height,
        )

        mask = scene.test_occlusions(rays).numpy()

        mask = np.where(mask == True, 1, 0).astype(np.uint8)
        return mask

        import cv2

        cv2.imshow("mask", mask * 255)
        cv2.waitKey(0)
        return mask

    def _calculate_occluded_mask(
        self, unoccluded_mask: np.ndarray, obj: LabelledObject, depth: np.ndarray
    ) -> np.ndarray:
        """by knowing what the depth image *should* look like, we can calculate the occluded mask"""
        # TODO implement this
        return unoccluded_mask
