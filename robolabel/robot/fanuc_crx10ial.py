from .robot import Robot, threadsafe
import numpy as np
import requests
import json
from scipy.spatial.transform import Rotation as R
import logging


class FanucCRX10iAL(Robot):
    ROBOT_IP = "10.162.12.203"

    def __init__(self):
        super().__init__(name="crx")

    def move_to(self, pose: np.ndarray, block=True):
        raise NotImplementedError("Implement move to for fanuc")

    @property
    @threadsafe
    def pose(self) -> np.ndarray:
        robot_http = "http://" + self.ROBOT_IP + "/KAREL/"
        url = robot_http + "remoteposition"
        try:
            req = requests.get(url, timeout=1.0)
        except requests.exceptions.Timeout:
            logging.warn(f"Can not reach robot: {self.name}")
            return super().pose
        jdict = json.loads(req.text)
        pose, _ = self.__parse_remote_position(jdict)
        self._pose = pose
        return self._pose

    @pose.setter
    def pose(self, pose: np.ndarray):
        logging.warn("Cant *set* pose for robot; use move_to instead")

    def __parse_remote_position(self, result):
        pose = np.array(
            [
                result["x"],
                result["y"],
                result["z"],
                result["w"],
                result["p"],
                result["r"],
            ]
        )
        joint_positions = [
            result["j1"],
            result["j2"],
            result["j3"],
            result["j4"],
            result["j5"],
            result["j6"],
        ]
        pose = self.__fanuc_6d_to_mat(pose)
        return pose, joint_positions

    def __fanuc_6d_to_mat(self, vec_6d):
        vec_6d[:3] = vec_6d[:3] / 1000.0
        vec_6d[3:] = vec_6d[3:] / 180 * np.pi

        orn = R.from_euler("xyz", vec_6d[3:]).as_matrix()
        pos = vec_6d[:3]
        return np.block([[orn, pos[:, None]], [np.zeros((1, 3)), 1]])

    def __mat_to_fanuc_6d(self, mat):
        pose6d = np.zeros((6,))
        pose6d[:3] = mat[:3, 3]
        pose6d[3:] = mat.from_matrix(mat[:3, :3]).as_euler("xyz")
        pose6d[:3] = pose6d[:3] * 1000
        pose6d[3:] = pose6d[3:] / np.pi * 180
        return pose6d
