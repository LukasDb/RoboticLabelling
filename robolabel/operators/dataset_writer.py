import asyncio
import json
import open3d as o3d
import numpy as np
import signal
import logging
from pathlib import Path
import PIL
from PIL import Image
import dataclasses
from dataclasses import dataclass


from ..labelled_object import LabelledObject
from ..camera import Camera, DepthQuality
from robolabel.lib.geometry import invert_homogeneous
from robolabel.lib.exr import EXR


class DelayedKeyboardInterrupt:
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
    occlusion_threshold: float = 0.03
    _is_pre_acquisition: bool = False


class DatasetWriter:
    def __init__(self) -> None:
        self._active_objects: list[LabelledObject] = []
        self._is_pre_acquisition: bool = True
        self.subdirs: dict[Camera, dict[str, Path]] = {}

        self.hq_depths: dict[Camera, dict[int, np.ndarray]] = {}
        self.hq_depths_R: dict[Camera, dict[int, np.ndarray | None]] = {}

        self.rendered_depths: dict[Camera, dict[int, np.ndarray]] = {}

    def setup(self, objects: list[LabelledObject], settings: WriterSettings | None) -> None:
        if settings is None:
            settings = WriterSettings()
        self._active_objects = objects
        self._output_dir = Path(settings.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._is_pre_acquisition = settings._is_pre_acquisition
        self.occlusion_threshold = settings.occlusion_threshold

        self._step: int | None = None
        if self._is_pre_acquisition:
            # clear cache of HQ depths
            self.hq_depths.clear()
            self.hq_depths_R.clear()

    async def capture(self, cam: Camera, idx_trajectory: int) -> None:
        """capture datapoint at trajectory point: idx_trajectory"""
        cam_dir = self._output_dir / cam.name

        if cam not in self.subdirs:
            self.subdirs[cam] = {
                k: cam_dir / k for k in ["gt", "mask", "rgb", "depth", "depth_HQ"]
            }

        for dir in self.subdirs[cam].values():
            dir.mkdir(parents=True, exist_ok=True)

        if self._step is None:
            # set the start step initially (to append)
            existing_images = list(self.subdirs[cam]["rgb"].glob("*"))
            existing_indices = [int(p.stem.split("_")[1]) for p in existing_images]
            if len(existing_indices) == 0:
                dataset_index = 0
            else:
                dataset_index = max(existing_indices) + 1
            self._step = dataset_index

        # handle keyboard interrupt or exceptions
        with DelayedKeyboardInterrupt():
            try:
                if not self._is_pre_acquisition:
                    await self._generate_data(cam, idx_trajectory)
                else:
                    self._acquire_pre_acquisition(cam, idx_trajectory)
            except Exception as e:
                self._cleanup(cam, self._step)
                raise e
            finally:
                self._step += 1

    def _acquire_pre_acquisition(self, cam: Camera, idx_trajectory: int):
        """acquires high-quality depth to be used for automatic segmentation labelling"""
        logging.debug(f"Acquiring pre-acquisition data for #{idx_trajectory}")
        frame = cam.get_frame(depth_quality=DepthQuality.GT)
        assert frame.depth is not None, "Depth frame is None"

        hq_depth = frame.depth
        hq_depth_R = frame.depth_R

        self.hq_depths.setdefault(cam, {})[idx_trajectory] = hq_depth
        self.hq_depths_R.setdefault(cam, {})[idx_trajectory] = hq_depth_R

    async def _generate_data(self, cam: Camera, idx_trajectory: int):
        """writes a datapoint with all GT labels"""
        logging.debug(f"Generating data #{self._step}")
        frame = cam.get_frame(depth_quality=DepthQuality.INFERENCE)
        cam_pos = await cam.get_position()
        cam_rot = await cam.get_orientation()
        try:
            hq_depth = self.hq_depths[cam][idx_trajectory]
            hq_depth_R = self.hq_depths_R[cam][idx_trajectory]
        except KeyError:
            logging.error(f"Missing pre-acquisition data for!")
            raise

        assert frame.rgb is not None, "RGB frame is None"
        assert frame.depth is not None, "Depth frame is None"

        # write RGB, RGB_R, depth, depth_R, depth_hq, depth_HQ_R to file
        rgb = Image.fromarray(frame.rgb)
        rgb.save(self.subdirs[cam]["rgb"] / f"rgb_{self._step:04}.png")

        with EXR(self.subdirs[cam]["depth"] / f"depth_{self._step:04}.exr") as F:
            F.write({"R": frame.depth.astype(np.float16)})

        if frame.rgb_R is not None:
            rgb_R = Image.fromarray(frame.rgb_R)
            rgb_R.save(self.subdirs[cam]["rgb"] / f"rgb_{self._step:04}_R.png")

        if frame.depth_R is not None:
            with EXR(self.subdirs[cam]["depth"] / f"depth_{self._step:04}_R.exr") as F:
                F.write({"R": frame.depth_R.astype(np.float16)})

        # write HQ depth to file
        with EXR(self.subdirs[cam]["depth_HQ"] / f"depth_{self._step:04}.exr") as F:
            F.write({"R": hq_depth.astype(np.float16)})

        if hq_depth_R is not None:
            with EXR(self.subdirs[cam]["depth_HQ"] / f"depth_{self._step:04}_R.exr") as F:
                F.write({"R": hq_depth_R.astype(np.float16)})

        # write masks
        masks: dict[str, np.ndarray] = {}
        visible_mask = np.zeros((cam.height, cam.width), dtype=np.uint8)
        for object_id, obj in enumerate(self._active_objects, start=1):
            unoccluded_mask = await self._render_object_mask(obj, cam)
            masks[f"{object_id:04}.R"] = unoccluded_mask.astype(np.float16)

            occluded_mask = await self._calculate_occluded_mask(
                unoccluded_mask, obj, cam, hq_depth
            )
            visible_mask[occluded_mask == 1] = object_id

        masks["visib.R"] = visible_mask.astype(np.float16)
        EXR(self.subdirs[cam]["mask"] / f"mask_{self._step:04}.exr").write(masks)

        # write GT metadata
        obj_list = []
        for object_id, obj in enumerate(self._active_objects, start=1):
            px_count_visib = np.count_nonzero(visible_mask == object_id)
            bbox_visib = self._get_bbox(visible_mask, object_id)

            obj_mask = masks[f"{object_id:04}.R"]
            bbox_obj = self._get_bbox(obj_mask, 1)
            px_count_all = np.count_nonzero(obj_mask == 1)
            px_count_valid = np.count_nonzero(frame.depth[visible_mask == object_id])
            visible_fraction = 0.0 if px_count_all == 0 else px_count_visib / px_count_all

            obj_list.append(
                {
                    "class": obj.name.split(".")[0],
                    "object id": object_id,
                    "pos": list(await obj.get_position()),
                    "rotation": list((await obj.get_orientation()).as_quat(canonical=True)),
                    "bbox_visib": bbox_visib,
                    "bbox_obj": bbox_obj,
                    "px_count_visib": px_count_visib,
                    "px_count_valid": px_count_valid,
                    "px_count_all": px_count_all,
                    "visib_fract": visible_fraction,
                }
            )

        cam_matrix = cam.intrinsic_matrix

        meta_dict = {
            "cam_rotation": list(cam_rot.as_quat(canonical=True)),
            "cam_location": list(cam_pos),
            "cam_matrix": np.array(cam_matrix).tolist(),
            "objs": list(obj_list),
        }

        with (self.subdirs[cam]["gt"] / f"gt_{self._step:05}.json").open("w") as F:
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

    def _cleanup(self, cam: Camera, dataset_index):
        for folder in self.subdirs[cam].values():
            for file in folder.glob(f"*_{dataset_index:04}.*"):
                logging.debug(f"Removing {file}")
                file.unlink()

    async def _render_object_mask(self, obj: LabelledObject, cam: Camera) -> np.ndarray:
        """render object mask from camera"""
        scene, rays = await self._get_raycasting_scene(obj, cam)
        mask = scene.test_occlusions(rays).numpy()
        mask = np.where(mask == True, 1, 0).astype(np.uint8)

        # import cv2
        # cv2.imshow("mask", mask * 255)
        # cv2.waitKey(0)
        # return mask
        return mask

    async def _calculate_occluded_mask(
        self,
        unoccluded_mask: np.ndarray,
        obj: LabelledObject,
        cam: Camera,
        depth: np.ndarray,
    ) -> np.ndarray:
        """by knowing what the depth image *should* look like, we can calculate the occluded mask"""

        scene, rays = await self._get_raycasting_scene(obj, cam)
        ans = scene.cast_rays(rays)
        rendered_depth = ans["t_hit"].numpy()

        diff = np.abs(rendered_depth - depth)

        occluded_mask = np.where(
            np.logical_and(diff < self.occlusion_threshold, unoccluded_mask == 1), 1, 0
        ).astype(np.uint8)

        # import cv2

        # color_depth = lambda x, scale=2.0: cv2.cvtColor(  # type: ignore
        #     cv2.applyColorMap(  # type: ignore
        #         cv2.convertScaleAbs(x, alpha=255 / scale), cv2.COLORMAP_JET  # type: ignore
        #     ),
        #     cv2.COLOR_BGR2RGB,  # type: ignore
        # )

        # prev = np.hstack([color_depth(depth), color_depth(rendered_depth)])
        # prev2 = np.hstack(
        #     [color_depth(diff, scale=0.5), np.stack([occluded_mask * 255] * 3, axis=-1)]
        # )
        # prev = np.vstack([prev, prev2])

        return occluded_mask

    async def _get_raycasting_scene(self, obj: LabelledObject, cam: Camera, visualize_debug=False):
        scene = o3d.t.geometry.RaycastingScene()

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh)
        mesh_t.transform(await obj.pose)
        mesh_id = scene.add_triangles(mesh_t)

        if visualize_debug:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                cam.width,
                cam.height,
                cam.intrinsic_matrix,
                invert_homogeneous(await cam.pose),
                1.0,
            )

            mesh = o3d.geometry.TriangleMesh(obj.mesh)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(np.array(obj.semantic_color) / 255.0)
            mesh.transform(await obj.pose)
            o3d.visualization.draw_geometries([mesh, frustum])  # type: ignore

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(cam.intrinsic_matrix),
            o3d.core.Tensor(invert_homogeneous(await cam.pose)),
            cam.width,
            cam.height,
        )

        return scene, rays
