from ..entity import Entity
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod


class TargetNotReachedError(Exception):
    pass


class Robot(Entity, ABC):
    def __init__(self, name: str, home_pose: npt.NDArray[np.float64] | None = None) -> None:
        Entity.__init__(self, name)
        self.home_pose = home_pose

    @property
    @abstractmethod
    async def pose(self) -> npt.NDArray[np.float64]:
        pass

    @pose.setter
    def pose(self, pose: npt.NDArray[np.float64]):
        self._pose = pose

    @abstractmethod
    async def set_current_as_homepose(self) -> None:
        pass

    @abstractmethod
    async def move_to(self, pose: npt.NDArray[np.float64], timeout: float):
        """move to a pose"""
        pass

    @abstractmethod
    async def stop(self):
        pass
