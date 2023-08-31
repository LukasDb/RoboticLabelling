from abc import ABC, abstractmethod, abstractproperty

from ..entity import Entity
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

        self.robot: Robot | None = None
        self._extrinsics: np.ndarray | None = None
        self._intrinsics: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None

    @property
    def pose(self) -> np.ndarray:
        if self.robot is None or self._extrinsics is None:
            pose = self._pose
        else:
            pose = self.robot.pose @ self._extrinsics
        return pose

    def attach(self, robot: Robot, link_matrix: np.ndarray):
        self.robot = robot
        self._extrinsics = link_matrix
        self._pose = self.robot.pose @ self._extrinsics
        self.notify(Event.CAMERA_ATTACHED)

    def detach(self):
        self.robot = None
        self._extrinsics = None

    @abstractproperty
    def unique_id(self) -> str:
        """Unique identifier for this camera. (such as serial number etc)"""
        return ""

    @property
    def intrinsic_matrix(self):
        return self._intrinsics

    @property
    def extrinsic_matrix(self):
        return self._extrinsics

    @property
    def dist_coeffs(self):
        return self._dist_coeffs

    def set_calibration(
        self,
        intrinsic_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        extrinsic_matrix: np.ndarray,
    ):
        self._intrinsics = intrinsic_matrix
        self._dist_coeffs = dist_coeffs
        self._extrinsics = extrinsic_matrix

        self.notify(Event.CAMERA_CALIBRATED)

    @abstractmethod
    def get_frame(self) -> CamFrame:
        pass
