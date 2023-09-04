from abc import ABC, abstractmethod, abstractproperty

from ..entity import Entity
from ..observer import Observable, Event
from ..robot import Robot
from dataclasses import dataclass
import numpy as np
from typing import Dict
from enum import Enum, auto


@dataclass
class CamFrame:
    rgb: np.ndarray | None = None
    rgb_R: np.ndarray | None = None
    depth: np.ndarray | None = None
    depth_R: np.ndarray | None = None


class DepthQuality(Enum):
    """Depth quality for the camera.
    INFERENCE should be used when the camera is used in production and for inference (fast, practical).
    GT should be used when the camera is used for ground truth generation (highest quality).
    FASTEST should be used when the camera is set to the fastest possible settings (probably lowest quality).
    """

    INFERENCE = auto()
    GT = auto()
    FASTEST = auto()
    UNCHANGED = auto()


class Camera(Entity, Observable, ABC):
    width: int
    height: int

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
        self.notify(Event.CAMERA_CALIBRATED)

    @abstractproperty
    def unique_id(self) -> str:
        """Unique identifier for this camera. (such as serial number etc)"""
        return ""

    @property
    def intrinsic_matrix(self):
        assert self._intrinsics is not None
        return self._intrinsics

    @property
    def extrinsic_matrix(self):
        assert self._extrinsics is not None
        return self._extrinsics

    @property
    def dist_coeffs(self):
        assert self._dist_coeffs is not None
        return self._dist_coeffs

    def is_calibrated(self):
        return (
            self._intrinsics is not None
            and self._dist_coeffs is not None
            and self._extrinsics is not None
        )

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
    def get_frame(self, depth_quality: DepthQuality) -> CamFrame:
        pass
