import asyncio
import logging
import numpy as np
from robolabel.robot.robot import Robot, threadsafe
import time


class MockRobot(Robot):
    def __init__(self):
        super().__init__(name="mock")

    @threadsafe
    def set_current_as_homepose(self) -> None:
        self.home_pose = self._pose

    @threadsafe
    async def move_to(self, pose: np.ndarray):
        await asyncio.sleep(0.5)
        self._pose = pose
        logging.info(f"{self.name} moved to pose: {pose[:3, 3]}")
