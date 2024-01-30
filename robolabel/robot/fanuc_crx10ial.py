from .robot import Robot, TargetNotReachedError
import asyncio

# import requests
import aiohttp
from robolabel.lib.geometry import distance_from_matrices
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import logging
import time

# TODO: Test move to

logging.getLogger("requests").setLevel(logging.WARNING)


class FanucCRX10iAL(Robot):
    ROBOT_IP = "192.168.1.100"

    def __init__(self) -> None:
        super().__init__(name="crx")
        self.session = aiohttp.ClientSession()

    async def move_to(self, pose: np.ndarray, timeout=20):
        t_started = time.time()
        await self.send_move(pose)
        while distance_from_matrices(await self.pose, pose) > 0.001:
            await asyncio.sleep(0.2)
            if time.time() - t_started > timeout:
                await self.stop()
                raise TargetNotReachedError(f"Move timed out after {timeout} seconds")

    async def stop(self):
        await self.send_move(await self.pose, interrupt=True)

    async def send_move(self, target_pose: np.ndarray, interrupt: bool = False):
        robot_http = "http://" + self.ROBOT_IP + "/KAREL/"
        url = robot_http + "remotemove"

        target_6d = self.__mat_to_fanuc_6d(target_pose)

        http_params = {
            "x": target_6d[0],
            "y": target_6d[1],
            "z": target_6d[2],
            "w": target_6d[3],
            "p": target_6d[4],
            "r": target_6d[5],
            "linear_path": 0,
            "interrupt": 1 if interrupt else 0,
        }

        # req = requests.get(url, params=http_params, timeout=10.0)
        req = await self.session.get(url, params=http_params)

        logging.debug(f"Answer: {req}")

    async def set_current_as_homepose(self) -> None:
        self.home_pose = await self.pose
        logging.info(f"Set robot {self.name}'s home pose to {self.home_pose[:3, 3]}")

    @property
    async def pose(self) -> np.ndarray:
        robot_http = "http://" + self.ROBOT_IP + "/KAREL/"
        url = robot_http + "remoteposition"
        # try:
        #     req = requests.get(url, timeout=10.0)
        # except requests.exceptions.ReadTimeout:
        #     return self._pose
        async with self.session.get(url) as resp:
            # data = await resp.json()
            text = await resp.text()
        # data = json.loads(req.text)

        try:
            data = json.loads(text)
        except Exception:
            print(resp)
            return await self.pose
        pose = self.__parse_remote_position(data)
        self._pose = pose
        return self._pose

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        raise ValueError("Cant *set* pose for robot; use move_to instead")

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
        # joint_positions = [
        #     result["j1"],
        #     result["j2"],
        #     result["j3"],
        #     result["j4"],
        #     result["j5"],
        #     result["j6"],
        # ]
        pose = self.__fanuc_6d_to_mat(pose)
        return pose

    def __fanuc_6d_to_mat(self, vec_6d):
        vec_6d[:3] = vec_6d[:3] / 1000.0
        vec_6d[3:] = vec_6d[3:] / 180 * np.pi

        mat = np.eye(4)
        mat[:3, :3] = R.from_euler("xyz", vec_6d[3:]).as_matrix()
        mat[:3, 3] = vec_6d[:3]
        return mat

    def __mat_to_fanuc_6d(self, mat):
        pose6d = np.zeros((6,))
        pose6d[:3] = mat[:3, 3]
        pose6d[3:] = R.from_matrix(mat[:3, :3]).as_euler("xyz")
        pose6d[:3] = pose6d[:3] * 1000
        pose6d[3:] = pose6d[3:] / np.pi * 180
        return pose6d
