import numpy as np
from ..entity import AsyncEntity
from scipy.spatial.transform import Rotation as R
import numpy.typing as npt
from abc import ABC, abstractmethod


class TargetNotReachedError(Exception):
    pass


class Robot(AsyncEntity, ABC):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @abstractmethod
    async def move_to(self, pose: npt.NDArray[np.float64], timeout: float) -> bool:
        """move to a pose"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass
