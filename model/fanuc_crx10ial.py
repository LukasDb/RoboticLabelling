from .robot import Robot
import numpy as np
import requests
import json
from scipy.spatial.transform import Rotation as R


class FanucCRX10iAL(Robot):
    ROBOT_IP = "10.162.12.203"

    def __init__(self):
        super().__init__(name="crx")

    def connect(self):
        """connect to the physical robot"""
        pass

    def disconnect(self):
        """disconnect from the physical robot"""
        pass

    def move_to(self, pose: np.ndarray, block=True):
        """move to a pose"""
        pass

    def move_to_joint(self, joint: np.ndarray, block=True):
        """move to a joint"""
        pass

    def get_position(self) -> np.ndarray:
        self.__update_pose()
        return super().get_position()

    def get_orientation(self) -> R:
        self.__update_pose()
        return super().get_orientation()

    def get_transform(self) -> np.ndarray:
        self.__update_pose()
        return super().get_transform()

    def get_joint(self) -> np.ndarray:
        """get current joint"""
        raise NotImplementedError("get join for fanuc not yet implemented")

    def __update_pose(self):
        robot_http = "http://" + self.ROBOT_IP + "/KAREL/"
        url = robot_http + "remoteposition"
        req = requests.get(url, timeout=1.0)
        jdict = json.loads(req.text)
        position, orientation, joint_positions = self.__parse_remote_position(jdict)
        self._position = position
        self._orientation = orientation

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
        position, orientation = self.__fanuc_6d_to_pos_orn(pose)
        return position, orientation, joint_positions

    def __fanuc_6d_to_pos_orn(self, vec_6d):
        vec_6d[:3] = vec_6d[:3] / 1000.0
        vec_6d[3:] = vec_6d[3:] / 180 * np.pi

        orn = R.from_euler("xyz", vec_6d[3:])
        pos = vec_6d[:3]
        return pos, orn

    def __pos_orn_to_fanuc_6d(self, pos, orn):
        pose6d = np.zeros((6,))
        pose6d[:3] = pos
        pose6d[3:] = orn.as_euler("xyz")
        pose6d[:3] = pose6d[:3] * 1000
        pose6d[3:] = pose6d[3:] / np.pi * 180
        return pose6d
