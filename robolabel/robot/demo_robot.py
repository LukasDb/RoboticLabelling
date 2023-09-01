import asyncio
import logging
import numpy as np
from robolabel.robot.robot import Robot
import time


class MockRobot(Robot):
    def __init__(self):
        super().__init__(name="mock")

    def set_current_as_homepose(self) -> None:
        self.home_pose = self._pose

    async def move_to(self, pose: np.ndarray) -> bool:
        await asyncio.sleep(0.5)
        self._pose = pose
        logging.info(f"{self.name} moved to pose: {pose[:3, 3]}")
        return True
