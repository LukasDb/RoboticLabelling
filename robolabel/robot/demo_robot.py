import asyncio
import logging
import numpy as np
from robolabel.robot.robot import Robot
import time


class MockRobot(Robot):
    def __init__(self) -> None:
        super().__init__(name="mock")

    async def stop(self) -> None:
        pass

    async def move_to(self, pose: np.ndarray, timeout: float = 1.0) -> bool:
        await asyncio.sleep(0.5)
        self._pose = pose
        logging.info(f"{self.name} moved to pose: {pose[:3, 3]}")
        return True
