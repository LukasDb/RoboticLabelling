import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from model.scene import Scene
from lib.geometry import invert_homogeneous

import time


class PoseRegistrator:
    # handles the initial object pose registration
    def __init__(self, scene: Scene):
        super().__init__()
        # TODO set initial pose to the center of the charuco board
        # TODO set background monitor so some 'easy' background
        # TODO set lighting to standard lighting
        self._scene = scene

    def optimize_pose(self):
        # TODO optimize pose in each image using ICP
        # TODO optimize object pose over all images
        # TODO inform user about optimization result
        # TODO save optimized pose to the scene
        monitor_pose = self._scene.background.pose
        for obj in self._scene.objects.values():
            obj.register_pose(monitor_pose)

    def draw_registered_objects(self, rgb, cam_pose, cam_intrinsics, cam_dist_coeffs):
        if cam_intrinsics is None:
            return rgb

        # get points of object mesh
        for obj in self._scene.objects.values():
            if not obj.registered:
                continue
            points = np.asarray(obj.mesh.vertices)
            cam2obj = invert_homogeneous(cam_pose) @ obj.pose
            rvec, _ = cv2.Rodrigues(cam2obj[:3, :3])
            tvec = cam2obj[:3, 3]

            projected_points, _ = cv2.projectPoints(
                points, rvec, tvec, cam_intrinsics, cam_dist_coeffs
            )
            projected_points = projected_points.astype(np.int32)
            rgb[projected_points[:, 0, 1], projected_points[:, 0, 0]] = obj.semantic_color

        return rgb
