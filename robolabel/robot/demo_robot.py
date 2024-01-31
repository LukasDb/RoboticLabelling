import asyncio
import logging
import numpy as np
from robolabel.robot.robot import Robot
import time


class MockRobot(Robot):
    def __init__(self) -> None:
        super().__init__(name="mock")
        self.home_pose = np.eye(4)

    async def set_current_as_homepose(self) -> None:
        self.home_pose = await self.get_pose()

    async def stop(self) -> None:
        pass

    async def move_to(self, pose: np.ndarray, timeout: float = 1.0) -> bool:
        await asyncio.sleep(0.5)
        self._pose = pose
        logging.info(f"{self.name} moved to pose: {pose[:3, 3]}")
        return True
