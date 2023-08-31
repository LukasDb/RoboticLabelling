from abc import ABC, abstractmethod, abstractproperty

from ..entity import Entity, threadsafe
from ..observer import Observable, Event
from ..robot import Robot
from dataclasses import dataclass
import numpy as np
from typing import Dict


@dataclass
class CamFrame:
    rgb: np.ndarray | None = None
    rgb_R: np.ndarray | None = None
    depth: np.ndarray | None = None
    depth_R: np.ndarray | None = None


class Camera(Entity, Observable, ABC):
    def __init__(self, name: str):
        Entity.__init__(self, name)
        Observable.__init__(self)

        self.robot: Robot = None
        self._link_matrix: np.ndarray = None

        self._intrinsics: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None

    @property
    @threadsafe
    def pose(self) -> np.ndarray:
        if self.robot is None:
            pose = self._pose
        else:
            pose = self.robot.pose @ self._link_matrix
        return pose

    @threadsafe
    def attach(self, robot: Robot, link_matrix: np.ndarray):
        self.robot = robot
        self._link_matrix = link_matrix
        self._pose = self.robot.pose @ self._link_matrix
        self.notify(Event.CAMERA_ATTACHED)

    @threadsafe
    def detach(self):
        self.robot = None
        self._link_matrix = None

    @abstractproperty
    def unique_id(self) -> str:
        """Unique identifier for this camera. (such as serial number etc)"""
        pass

    @property
    def intrinsic_matrix(self):
        return self._intrinsics

    @property
    def extrinsic_matrix(self):
        return self._link_matrix

    @property
    def dist_coeffs(self):
        return self._dist_coeffs

    @threadsafe
    def set_calibration(
        self,
        intrinsic_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        extrinsic_matrix: np.ndarray,
    ):
        self._intrinsics = intrinsic_matrix
        self._dist_coeffs = dist_coeffs
        self._link_matrix = extrinsic_matrix

        self.notify(Event.CAMERA_CALIBRATED)

    @abstractmethod
    def get_frame(self) -> CamFrame:
        pass
