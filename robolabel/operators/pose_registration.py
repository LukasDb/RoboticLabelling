import numpy as np
import numpy.typing as npt
import open3d as o3d
import asyncio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import cv2
from dataclasses import dataclass
from scipy import optimize
import logging

import robolabel as rl
from robolabel.operators import TrajectoryGenerator, TrajectorySettings, Acquisition
from robolabel.observer import Event
from robolabel.lib.geometry import (
    invert_homogeneous,
    get_affine_matrix_from_euler,
)


@dataclass
class Datapoint:
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    intrinsics: np.ndarray
    dist_coeffs: np.ndarray


class PoseRegistration(rl.Observer):
    # handles the initial object pose registration
    def __init__(self, scene: rl.Scene) -> None:
        super().__init__()
        # TODO set lighting to standard lighting
        self._scene = scene
        self.listen_to(self._scene)
        self.is_active = False
        self.mesh_cache = {}
        self.datapoints: list[Datapoint] = []
        self.trajectory_generator = TrajectoryGenerator()
        self.acquisition = Acquisition()

    def update_observer(self, subject: rl.Observable, event: Event, *args, **kwargs):
        if event == Event.MODE_CHANGED:
            if kwargs["mode"] == "registration":
                self.reset()
                self.is_active = True
                self._scene.background.set_textured()  # easier to detect -> less noise
            else:
                self.is_active = False

    def reset(self) -> None:
        self.datapoints.clear()

    async def capture(self) -> Datapoint | None:
        if self._scene.selected_camera is None:
            logging.error("No camera selected")
            return None

        # best settings
        cam_pose = await self._scene.selected_camera.pose
        frame = self._scene.selected_camera.get_frame(depth_quality=rl.camera.DepthQuality.GT)
        if frame.rgb is None or not self._scene.selected_camera.is_calibrated():
            return None

        if frame.depth is None:
            logging.error("Camera must have a depth channel")
            return

        datapoint = Datapoint(
            rgb=frame.rgb,
            depth=frame.depth,
            pose=cam_pose,
            intrinsics=self._scene.selected_camera.intrinsic_matrix,
            dist_coeffs=self._scene.selected_camera.dist_coefficients,
        )

        self.datapoints.append(datapoint)
        return datapoint

    async def capture_images(self) -> None:
        if self._scene.selected_camera is None:
            logging.error("No camera selected")
            return None

        trajectory_settings = TrajectorySettings(
            n_steps=20, view_jitter=0.0, z_cutoff=0.3, r_range=(0.35, 0.4), roll_range=(0, 0)
        )
        obj = self._scene.selected_object
        if obj is None:
            logging.error("No object selected")
            return
        center = (await obj.pose)[:3, 3]

        trajectory = self.trajectory_generator.generate_trajectory_above_center(
            center, trajectory_settings
        )

        # await self.trajectory_generator.visualize_trajectory(self._scene.selected_camera, [])

        logging.debug("Starting acquisition for calibration...")

        i = 0
        async for _ in self.acquisition.execute([self._scene.selected_camera], trajectory):
            i += 1
            logging.debug(f"Reached {i}/{len(trajectory)}")
            await self.capture()

    @rl.as_async_task
    async def optimize_pose(self) -> None:
        obj = self._scene.selected_object
        if obj is None:
            logging.error("No object selected")
            return
        obj.mesh.compute_vertex_normals()
        obj_points = obj.mesh.sample_points_poisson_disk(1000)

        valid_campose_icp = []
        obj_pose = await obj.pose
        for datapoint in tqdm(self.datapoints, desc="ICP"):
            initial_guess = invert_homogeneous(datapoint.pose) @ obj_pose
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=datapoint.depth.shape[1],
                height=datapoint.depth.shape[0],
                intrinsic_matrix=datapoint.intrinsics,
            )
            target_pcl = o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(datapoint.depth),
                intrinsics,
                np.eye(4),  # extrinsic
                depth_scale=1.0,
                depth_trunc=2.0,
                stride=1,
                project_valid_depth_only=True,
            )
            target_pcl.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            target_pcl.orient_normals_towards_camera_location()

            result = o3d.pipelines.registration.registration_icp(
                obj_points,
                target_pcl,
                0.02,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # TransformationEstimationPointToPlane(),
            )

            if result.fitness < 0.5:
                continue

            icp_RT = result.transformation  # ICP is in camera frame
            valid_campose_icp.append((datapoint.pose, icp_RT))

        valid_frac = len(valid_campose_icp) / len(self.datapoints)

        logging.info(
            f"Using {len(valid_campose_icp)}/{len(self.datapoints)} ({valid_frac*100:.1f}%) datapoints for optimization."
        )

        ret = self._optimize_object_pose(
            obj_pose,
            [x[0] for x in valid_campose_icp],
            [x[1] for x in valid_campose_icp],
        )

        logging.info("Done")
        logging.info(f"Optimality: {ret['optimality']}")
        logging.info(f"Cost:       {ret['cost']}")
        x = ret["x"]
        world2object = x.reshape((4, 4))
        logging.info(f"Result:\n{world2object}")

        obj.register_pose(world2object)

    def move_pose(self, obj: rl.LabelledObject, x, y, z, rho, phi, theta):
        obj.pose = get_affine_matrix_from_euler([rho, phi, theta], [x, y, z])

    async def get_from_image_cache(self, index):
        try:
            datapoint: Datapoint = self.datapoints[index]
        except IndexError:
            return None

        img = datapoint.rgb.copy()
        if self._scene.selected_object is not None:
            img = await self.draw_registered_object(
                self._scene.selected_object,
                img,
                datapoint.pose,
                datapoint.intrinsics,
                datapoint.dist_coeffs,
            )

        return img

    async def draw_on_preview(self, cam: rl.camera.Camera, rgb: np.ndarray):
        if not self.is_active or not cam.is_calibrated():
            return rgb

        for obj in self._scene.objects.values():
            await self.draw_registered_object(
                obj, rgb, await cam.pose, cam.intrinsic_matrix, cam.dist_coefficients
            )
        return rgb

    async def draw_registered_object(
        self,
        obj: rl.LabelledObject,
        rgb: np.ndarray,
        cam_pose: np.ndarray,
        cam_intrinsics: np.ndarray,
        cam_dist_coeffs: np.ndarray,
    ):
        cam2obj = invert_homogeneous(cam_pose) @ (await obj.pose)
        rvec, _ = cv2.Rodrigues(cam2obj[:3, :3])  # type: ignore
        tvec = cam2obj[:3, 3]

        # draw frame
        rgb = cv2.drawFrameAxes(  # type: ignore
            rgb, cam_intrinsics, cam_dist_coeffs, rvec, tvec, 0.1, 3
        )

        if obj.name not in self.mesh_cache:
            points = np.asarray(obj.mesh.vertices)
            points = np.asarray(obj.mesh.sample_points_poisson_disk(500).points)
            self.mesh_cache[obj.name] = points
        else:
            points = self.mesh_cache[obj.name]

        projected_points, _ = cv2.projectPoints(  # type: ignore
            points, rvec, tvec, cam_intrinsics, cam_dist_coeffs
        )
        projected_points = projected_points.astype(int)[:, 0, :]

        outside_x = np.logical_or(
            projected_points[:, 0] < 0, projected_points[:, 0] >= rgb.shape[1]
        )
        outside_y = np.logical_or(
            projected_points[:, 1] < 0, projected_points[:, 1] >= rgb.shape[0]
        )
        outside = np.logical_or(outside_x, outside_y)

        if np.count_nonzero(outside) / len(outside) > 0.8:  # 80% outside
            self._draw_tracking_arrow(
                rgb, rvec, tvec, cam_intrinsics, cam_dist_coeffs, obj.semantic_color
            )

        # clip to image size
        projected_points = np.clip(projected_points, 0, np.array(rgb.shape[1::-1]) - 1)

        for point in projected_points:
            cv2.circle(rgb, tuple(point), 4, obj.semantic_color, -1)  # type: ignore
        return rgb

    def _draw_tracking_arrow(
        self, rgb: np.ndarray, rvec, tvec, cam_intrinsics, cam_dist_coeffs, color
    ):
        centroid = np.array([0.0, 0.0, 0.0])
        projected_center = cv2.projectPoints(  # type: ignore
            centroid, rvec, tvec, cam_intrinsics, cam_dist_coeffs
        )[0][0][0]
        # dist to image border
        half_size = np.array((rgb.shape[1], rgb.shape[0])) / 2
        centroid_centered = projected_center - half_size
        outside_dist = np.clip(np.abs(centroid_centered) - half_size, 0, None)
        outside_dist = np.linalg.norm(outside_dist)  # used for thickness of arrow

        arrow_thickness = np.clip(int(5000 / outside_dist), 2, 10)

        dir_vec = centroid_centered / np.linalg.norm(centroid_centered)

        # find the point on the image border
        dist = np.linalg.norm(centroid_centered)
        scale = min(np.abs(half_size / centroid_centered))
        length_2_img_border = dist * scale

        arrow_length = 200
        arrow_start = (
            int(rgb.shape[1] / 2 + dir_vec[0] * (length_2_img_border - arrow_length)),
            int(rgb.shape[0] / 2 + dir_vec[1] * (length_2_img_border - arrow_length)),
        )
        arrow_end = (
            int(rgb.shape[1] / 2 + dir_vec[0] * length_2_img_border),
            int(rgb.shape[0] / 2 + dir_vec[1] * length_2_img_border),
        )
        cv2.arrowedLine(rgb, arrow_start, arrow_end, color, arrow_thickness)  # type: ignore

    def _optimize_object_pose(self, world2object, world2camera, camera2object):
        def residual(world2object_, world2camera, camera2object):
            res: list[npt.NDArray[np.float64]] = []
            world2object = world2object_.reshape((4, 4))
            for i in range(len(world2camera)):
                res.append(single_res_func(world2camera[i], camera2object[i], world2object))
            return np.array(res).reshape(16 * len(world2camera)) / len(world2camera)

        def single_res_func(world2camera, camera2object, world2object):
            res_array = world2camera @ camera2object - world2object
            return res_array.reshape((16,))

        ret = optimize.least_squares(
            residual,
            world2object.reshape((16,)),
            kwargs={"world2camera": world2camera, "camera2object": camera2object},
        )
        return ret
