from ..entity import Entity, threadsafe
import numpy as np
from abc import ABC, abstractmethod


class Robot(Entity, ABC):
    def __init__(self, name: str):
        Entity.__init__(self, name)

    @abstractmethod
    def move_to(self, pose: np.ndarray, block=True):
        """move to a pose"""
        pass