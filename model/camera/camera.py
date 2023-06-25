from abc import ABC, abstractmethod, abstractproperty
from ..entity import Entity
from ..observer import Subject, Event
from dataclasses import dataclass
import numpy as np
from typing import Dict


@dataclass
class CamFrame:
    rgb: np.ndarray | None = None
    rgb_R: np.ndarray | None = None
    depth: np.ndarray | None = None
    depth_R: np.ndarray | None = None


class Camera(Entity, Subject, ABC):
    def __init__(self, name: str):
        Entity.__init__(self, name)
        Subject.__init__(self)
        self._intrinsics: np.ndarray | None = None
        self._extrinsics: np.ndarray = np.eye(4)
        self._dist_coeffs: np.ndarray | None = None

    @abstractproperty
    def unique_id(self) -> str:
        """Unique identifier for this camera. (such as serial number etc)"""
        pass

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
        with self.lock:
            self._intrinsics = intrinsic_matrix
            self._dist_coeffs = dist_coeffs
            self._extrinsics = extrinsic_matrix
        self.notify(Event.CAMERA_CALIBRATED)


    @abstractmethod
    def get_frame(self) -> CamFrame:
        pass
