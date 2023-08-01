import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
from dataclasses import dataclass

from model.scene import Scene
from lib.geometry import invert_homogeneous,get_affine_matrix_from_r_t

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
    def __init__(self, scene: Scene):
        super().__init__()
        # TODO set initial pose to the center of the charuco board
        # TODO set background monitor so some 'easy' background
        # TODO set lighting to standard lighting
        self._scene = scene

        self.datapoints: List[Datapoint] = []

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

    def optimize_pose(self):
        # TODO optimize pose in each image using ICP
        # TODO optimize object pose over all images
        # TODO inform user about optimization result
        # TODO save optimized pose to the scene
        monitor_pose = self._scene.background.pose
        for obj in self._scene.objects.values():
            obj.register_pose(monitor_pose)

    def move_pose(self,obj,x,y,z,rho,phi,theta):
        #obj.pose[0,3] = x
        #obj.pose[1,3] = y
        #obj.pose[2,3] = z
        obj.pose = get_affine_matrix_from_r_t([rho,phi,theta],[x,y,z])
        


    def draw_registered_objects(self, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs):
        for obj in self._scene.objects.values():
            if not obj.registered:
                continue
            self.draw_registered_object(
                obj, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs
            )
        return rgb

    def draw_registered_object(
        self, obj, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs
    ):
        if cam_intrinsics is None:
            return rgb

        points = np.asarray(obj.mesh.vertices)
        cam2obj = invert_homogeneous(cam_pose) @ obj.pose
        rvec, _ = cv2.Rodrigues(cam2obj[:3, :3])
        tvec = cam2obj[:3, 3]

        projected_points, _ = cv2.projectPoints(
            points, rvec, tvec, cam_intrinsics, cam_dist_coeffs
        )
        projected_points = projected_points.astype(np.int32)

        # clip to image size
        projected_points = np.clip(projected_points, 0, np.array(rgb.shape[1::-1]) - 1)

        rgb[projected_points[:, 0, 1], projected_points[:, 0, 0]] = obj.semantic_color

        return rgb
