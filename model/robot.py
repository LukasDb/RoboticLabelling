from .entity import Entity
import numpy as np
from abc import ABC, abstractmethod


class Robot(Entity, ABC):
    def __init__(self):
        Entity.__init__(self)
        self.is_online = False

    @abstractmethod
    def connect(self):
        """connect to the physical robot"""
        pass

    @abstractmethod
    def disconnect(self):
        """disconnect from the physical robot"""
        pass

    @abstractmethod
    def move_to(self, pose: np.ndarray, block=True):
        """move to a pose"""
        pass

    @abstractmethod
    def move_to_joint(self, joint: np.ndarray, block=True):
        """move to a joint"""
        pass

    @abstractmethod
    def get_pose(self) -> np.ndarray:
        """get current pose"""
        pass

    @abstractmethod
    def get_joint(self) -> np.ndarray:
        """get current joint"""
        pass
