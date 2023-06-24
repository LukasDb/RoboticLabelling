from abc import ABC, abstractmethod, abstractproperty
from ..entity import Entity
from dataclasses import dataclass
import numpy as np
from typing import Dict


@dataclass
class CamFrame:
    rgb: np.ndarray | None = None
    rgb_R: np.ndarray | None = None
    depth: np.ndarray | None = None
    depth_R: np.ndarray | None = None


class Camera(Entity, ABC):
    def __init__(self, name: str):
        Entity.__init__(self, name)
        self._intrinsics: np.ndarray | None = None
        self._extrinsics: np.ndarray | None = None
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

    # setter for intrinsic_matrix and dist_coeffs
    @intrinsic_matrix.setter
    def intrinsic_matrix(self, value: np.ndarray):
        self._intrinsics = value

    @dist_coeffs.setter
    def dist_coeffs(self, value: np.ndarray):
        self._dist_coeffs = value

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, value: np.ndarray):
        self._link_matrix = value
        self._extrinsics = value

    @abstractmethod
    def get_frame(self) -> CamFrame:
        pass
