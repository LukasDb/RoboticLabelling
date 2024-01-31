import json
import open3d as o3d
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
import simpose as sp
import contextlib


from ..labelled_object import LabelledObject
from ..camera import Camera, DepthQuality
from .trajectory_executor import TrajectoryExecutor
from .trajectory_generator import TrajectoryGenerator
from .background_monitor import BackgroundSettings
from .lights_controller import LightsSettings
from robolabel.geometry import invert_homogeneous
import robolabel as rl


@dataclass
class AcquisitionSettings:
    is_dry_run: bool = False
    output_dir: str = "dataset"
    occlusion_threshold: float = 0.03
    is_pre_acquisition: bool = False


# TODO change everything for sp.Writer


class DataAcquisition:
    def __init__(self, scene: rl.Scene) -> None:
        self.scene = scene
        self.trajectory_generator = TrajectoryGenerator()
        self.trajectory_executor = TrajectoryExecutor(scene)

        self.hq_depths: dict[Camera, dict[int, np.ndarray]] = {}
        self.hq_depths_R: dict[Camera, dict[int, np.ndarray | None]] = {}

    async def run(
        self,
        *,
        acquisition_settings: AcquisitionSettings,
        active_cameras: list[rl.camera.Camera],
        active_objects: list[LabelledObject],
        bg_settings: BackgroundSettings,
        lights_settings: LightsSettings,
    ) -> None:
        trajectory = self.trajectory_generator.get_current_trajectory()

        bg_monitor = self.scene.background
        lights_controller = self.scene.lights

        num_steps_per_camera = len(trajectory) * bg_settings.n_steps * lights_settings.n_steps

        if acquisition_settings.is_pre_acquisition:
            # clear cache of HQ depths
            self.hq_depths.clear()
            self.hq_depths_R.clear()

        if not acquisition_settings.is_pre_acquisition and not acquisition_settings.is_dry_run:
            assert num_steps_per_camera == len(
                self.hq_depths[active_cameras[0]]
            ), f"Missing HQ depth, run pre-acquisition first"

        writers: dict[rl.camera.Camera, sp.writers.Writer] = {}
        for cam in active_cameras:
            output_dir = Path(acquisition_settings.output_dir).expanduser().resolve()

            if output_dir.exists():
                raise ValueError(f"Output directory {output_dir} already exists")

            p = sp.writers.WriterConfig(
                output_dir=output_dir.with_name(f"{output_dir.name}_{cam.unique_id}"),
                overwrite=True,
                start_index=0,
                end_index=num_steps_per_camera - 1,
            )
            writers[cam] = sp.writers.TFRecordWriter(p, comm=None)

        with contextlib.ExitStack() as stack:
            open_writers: dict[rl.camera.Camera, sp.writers.Writer] = {
                cam: stack.enter_context(writer) for cam, writer in writers.items()
            }

            async for idx_trajectory, cam in self.trajectory_executor.execute(
                active_cameras,
                trajectory,
                bg_settings=bg_settings,
                lights_settings=lights_settings,
            ):
                logging.info(f"Reached {idx_trajectory+1}/{len(trajectory)} point in trajectory")
                if acquisition_settings.is_dry_run:
                    continue

                if acquisition_settings.is_pre_acquisition:
                    # store HQ depth for later
                    frame = cam.get_frame(depth_quality=DepthQuality.GT)

                    assert frame.depth is not None
                    hq_depth = frame.depth
                    hq_depth_R = frame.depth_R

                    self.hq_depths.setdefault(cam, {})[idx_trajectory] = hq_depth
                    self.hq_depths_R.setdefault(cam, {})[idx_trajectory] = hq_depth_R
                    continue

                frame = cam.get_frame(depth_quality=rl.camera.DepthQuality.INFERENCE)
                cam_pos = await cam.get_position()
                cam_rot = await cam.get_orientation()
                assert frame.depth is not None, "Depth image is None"
                hq_depth = self.hq_depths[cam][idx_trajectory]
                hq_depth_R = self.hq_depths_R[cam][idx_trajectory]

                # TODO get optional stereo baseline of camera

                # write masks

                object_labels: list[sp.ObjectAnnotation] = []
                visible_mask = np.zeros((cam.height, cam.width), dtype=np.uint8)

                for object_id, obj in enumerate(active_objects):
                    unoccluded_mask = await self.render_object_mask(obj, cam)

                    occluded_mask = await self.calculate_occluded_mask(
                        unoccluded_mask=unoccluded_mask,
                        obj=obj,
                        cam=cam,
                        depth=hq_depth,
                        occlusion_threshold=acquisition_settings.occlusion_threshold,
                    )
                    visible_mask[occluded_mask == 1] = object_id
                    px_count_visib = np.count_nonzero(visible_mask == object_id)
                    bbox_visib = self.get_bbox(visible_mask, object_id)

                    obj_mask = occluded_mask
                    bbox_obj = self.get_bbox(obj_mask, 1)
                    px_count_all = np.count_nonzero(obj_mask == 1)
                    px_count_valid = np.count_nonzero(frame.depth[visible_mask == object_id])
                    visible_fraction = 0.0 if px_count_all == 0 else px_count_visib / px_count_all

                    object_labels.append(
                        sp.ObjectAnnotation(
                            cls="cls",
                            object_id=object_id,
                            position=obj.get_position(),
                            quat_xyzw=obj.get_orientation().as_quat(canonical=True),
                            bbox_visib=bbox_visib,
                            bbox_obj=bbox_obj,
                            px_count_visib=px_count_visib,
                            px_count_valid=px_count_valid,
                            px_count_all=px_count_all,
                            visib_fract=visible_fraction,
                        )
                    )

                datapoint = sp.RenderProduct(
                    rgb=frame.rgb,
                    rgb_R=frame.rgb_R,
                    depth=frame.depth,
                    depth_R=frame.depth_R,
                    mask=visible_mask,
                    cam_position=cam_pos,
                    cam_rotation=cam_rot.as_quat(),
                    intrinsics=cam.intrinsic_matrix,
                    object_annotations=object_labels,
                )
                open_writers[cam].write_data(idx_trajectory, render_product=datapoint)

    def get_bbox(self, mask: np.ndarray, object_id: int) -> tuple[int, int, int, int]:
        y, x = np.where(mask == object_id)
        if len(y) == 0:
            return (0, 0, 0, 0)
        x1 = np.min(x).tolist()
        x2 = np.max(x).tolist()
        y1 = np.min(y).tolist()
        y2 = np.max(y).tolist()
        return (x1, y1, x2, y2)

    async def render_object_mask(self, obj: LabelledObject, cam: Camera) -> np.ndarray:
        """render object mask from camera"""
        scene, rays = await self.get_raycasting_scene(obj, cam)
        o3d_mask = scene.test_occlusions(rays).numpy()
        return np.where(o3d_mask == True, 1, 0).astype(np.uint8)

    async def calculate_occluded_mask(
        self,
        unoccluded_mask: np.ndarray,
        obj: LabelledObject,
        cam: Camera,
        depth: np.ndarray,
        occlusion_threshold: float,
    ) -> np.ndarray:
        """by knowing what the depth image *should* look like, we can calculate the occluded mask"""

        scene, rays = await self.get_raycasting_scene(obj, cam)
        ans = scene.cast_rays(rays)
        rendered_depth = ans["t_hit"].numpy()

        diff = np.abs(rendered_depth - depth)

        occluded_mask = np.where(
            np.logical_and(diff < occlusion_threshold, unoccluded_mask == 1), 1, 0
        ).astype(np.uint8)

        return occluded_mask

    async def get_raycasting_scene(
        self, obj: LabelledObject, cam: Camera, visualize_debug: bool = False
    ) -> tuple[o3d.t.geometry.RaycastingScene, o3d.core.Tensor]:
        scene = o3d.t.geometry.RaycastingScene()

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh)
        mesh_t.transform(obj.get_pose())
        mesh_id = scene.add_triangles(mesh_t)

        if visualize_debug:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                cam.width,
                cam.height,
                cam.intrinsic_matrix,
                invert_homogeneous(await cam.get_pose()),
                1.0,
            )

            mesh = o3d.geometry.TriangleMesh(obj.mesh)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(np.array(obj.semantic_color) / 255.0)
            mesh.transform(obj.get_pose())
            o3d.visualization.draw_geometries([mesh, frustum])  # type: ignore

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(cam.intrinsic_matrix),
            o3d.core.Tensor(invert_homogeneous(await cam.get_pose())),
            cam.width,
            cam.height,
        )

        return scene, rays
