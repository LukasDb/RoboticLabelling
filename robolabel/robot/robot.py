import asyncio
from ..entity import Entity, threadsafe
import numpy as np
import logging
from abc import ABC, abstractmethod


class Robot(Entity, ABC):
    def __init__(self, name: str, home_pose: np.ndarray | None = None) -> None:
        Entity.__init__(self, name)
        self.home_pose = home_pose

    @abstractmethod
    def set_current_as_homepose(self) -> None:
        pass

    @abstractmethod
    async def move_to(self, pose: np.ndarray) -> None:
        """move to a pose"""
        pass
