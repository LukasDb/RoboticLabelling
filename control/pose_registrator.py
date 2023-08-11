import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
from dataclasses import dataclass

from model.scene import Scene
from model.labelled_object  import LabelledObject
from lib.geometry import invert_homogeneous,get_affine_matrix_from_euler

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
        # save camera pose to file
        folder = "demo_data"
        index = len(self.datapoints)+10
        pose_path = (f"{folder}/poses/{self._scene.selected_camera}/{index}.txt")
        if not os.path.exists(os.path.dirname(pose_path)):
            os.makedirs(os.path.dirname(pose_path))
        np.savetxt(pose_path, datapoint.pose)
        # save rgb image and depth to file
        rgb_path = (f"{folder}/rgb/{self._scene.selected_camera}/{index}.png")
        if not os.path.exists(os.path.dirname(rgb_path)):
            os.makedirs(os.path.dirname(rgb_path))
        cv2.imwrite(rgb_path, cv2.cvtColor(datapoint.rgb, cv2.COLOR_RGB2BGR))
        if datapoint.depth is not None:
            depth_path = (f"{folder}/depth/{self._scene.selected_camera}/{index}.npz")  
            if not os.path.exists(os.path.dirname(depth_path)):
                os.makedirs(os.path.dirname(depth_path))
            np.save(depth_path, datapoint.depth)

        self.datapoints.append(datapoint)
        return datapoint

    def optimize_pose(self):
        # TODO optimize pose in each image using ICP
        # TODO optimize object pose over all images
        # TODO inform user about optimization result
        # TODO save optimized pose to the scene
        monitor_pose = self._scene.background.pose
        monitor_points = monitor_pose @ np.asarray(obj.mesh.vertices).T
        for obj in self._scene.objects.values():
            points = np.asarray(obj.mesh.vertices)
            optimized_pose=o3d.pipelines.registration.registration_icp(
                points,monitor_points)
            #source, target, threshold, trans_init,
            #o3d.registration.TransformationEstimationPointToPoint())
            obj.register_pose(optimized_pose)


    def move_pose(self,obj,x,y,z,rho,phi,theta):
        obj.pose = get_affine_matrix_from_euler([rho,phi,theta],[x,y,z])
        


    def draw_registered_objects(self, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs):
        for obj in self._scene.objects.values():
            if not obj.registered:
                continue
            self.draw_registered_object(
                obj, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs
            )
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
        projected_points = projected_points.astype(np.int32)

        # clip to image size
        projected_points = np.clip(projected_points, 0, np.array(rgb.shape[1::-1]) - 1)

        #rgb[projected_points[:, 0, 1], projected_points[:, 0, 0]] = obj.semantic_color
        for point in projected_points:
            cv2.circle(rgb, tuple(point[0]), 4, obj.semantic_color, -1)
        return rgb
