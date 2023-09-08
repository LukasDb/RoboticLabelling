import asyncio
import logging
import numpy as np
from robolabel.robot.robot import Robot
import time


class MockRobot(Robot):
    def __init__(self):
        super().__init__(name="mock")

    async def set_current_as_homepose(self) -> None:
        self.home_pose = await self.pose

    async def stop(self):
        pass

    @property
    async def pose(self) -> np.ndarray:
        return self._pose

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        self._pose = pose

    async def move_to(self, pose: np.ndarray, timeout: float = 1.0):
        await asyncio.sleep(0.5)
        self._pose = pose
        logging.info(f"{self.name} moved to pose: {pose[:3, 3]}")
