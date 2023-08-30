import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
from dataclasses import dataclass
from scipy import optimize
import logging

from robolabel.scene import Scene
from robolabel.labelled_object import LabelledObject
from robolabel.lib.geometry import (
    invert_homogeneous,
    get_affine_matrix_from_euler,
)

import time


@dataclass
class Datapoint:
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    intrinsics: np.ndarray
    dist_coeffs: np.ndarray


class PoseRegistrator:
    # handles the initial object pose registration
    def __init__(self, scene: Scene) -> None:
        super().__init__()
        # TODO set background monitor so some 'easy' background
        # TODO set lighting to standard lighting
        self._scene = scene

        self.datapoints: List[Datapoint] = []

    def reset(self) -> None:
        self.datapoints.clear()

    def capture_image(self) -> Datapoint | None:
        frame = self._scene.selected_camera.get_frame()
        if frame.rgb is None or self._scene.selected_camera.intrinsic_matrix is None:
            return None

        datapoint = Datapoint(
            rgb=frame.rgb,
            depth=frame.depth,
            pose=self._scene.selected_camera.pose,
            intrinsics=self._scene.selected_camera.intrinsic_matrix,
            dist_coeffs=self._scene.selected_camera.dist_coeffs,
        )

        self.datapoints.append(datapoint)
        return datapoint

    def optimize_pose(self) -> None:
        obj = self._scene.selected_object
        obj.mesh.compute_vertex_normals()
        obj_points = obj.mesh.sample_points_poisson_disk(1000)

        valid_campose_icp = []
        for datapoint in tqdm(self.datapoints, desc="ICP"):
            initial_guess = np.linalg.inv(datapoint.pose) @ obj.pose
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
            obj.pose,
            [x[0] for x in valid_campose_icp],
            [x[1] for x in valid_campose_icp],
        )

        logging.info("Done")
        logging.info(f"Optimality: {ret['optimality']}")
        logging.info(f"Cost:       {ret['cost']}")
        x = ret["x"]
        world2object = x.reshape((4, 4))
        logging.info(f"Result:\n{world2object}")

        self._scene.selected_object.register_pose(world2object)

    def move_pose(self, obj: LabelledObject, x, y, z, rho, phi, theta):
        obj.pose = get_affine_matrix_from_euler([rho, phi, theta], [x, y, z])

    def draw_registered_objects(self, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs):
        for obj in self._scene.objects.values():
            if not obj.registered:
                continue
            self.draw_registered_object(obj, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs)
        return rgb

    def draw_registered_object(
        self, obj: LabelledObject, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs
    ):
        if cam_intrinsics is None:
            return rgb

        points = np.asarray(obj.mesh.vertices)
        points = np.asarray(obj.mesh.sample_points_poisson_disk(500).points)
        cam2obj = invert_homogeneous(cam_pose) @ obj.pose
        rvec, _ = cv2.Rodrigues(cam2obj[:3, :3])
        tvec = cam2obj[:3, 3]

        projected_points, _ = cv2.projectPoints(
            points, rvec, tvec, cam_intrinsics, cam_dist_coeffs
        )
        projected_points = projected_points.astype(int)

        # clip to image size
        projected_points = np.clip(projected_points, 0, np.array(rgb.shape[1::-1]) - 1)

        for point in projected_points:
            cv2.circle(rgb, tuple(point[0]), 4, obj.semantic_color, -1)
        return rgb

    def _optimize_object_pose(self, world2object, world2camera, camera2object):
        def residual(world2object_, world2camera, camera2object):
            res = []
            world2object = world2object_.reshape((4, 4))
            for i in range(len(world2camera)):
                res += single_res_func(world2camera[i], camera2object[i], world2object)
            return np.array(res).reshape(16 * len(world2camera))

        def single_res_func(world2camera, camera2object, world2object):
            res_array = world2camera @ camera2object - world2object
            return [res_array.reshape((16,))]

        ret = optimize.least_squares(
            residual,
            world2object.reshape((16,)),
            kwargs={"world2camera": world2camera, "camera2object": camera2object},
        )
        return ret
