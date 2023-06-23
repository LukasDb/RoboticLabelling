from abc import ABC
from ..entity import Entity
from dataclasses import dataclass
import numpy as np
from typing import Dict


@dataclass
class CamFrame:
    rgb: np.ndarray | None = None
    rgb_R: np.ndarray | None = None
    depth_HQ: np.ndarray | None = None
    depth_LQ: np.ndarray | None = None
    extrinsic: np.ndarray | None = None
    intrinsic: np.ndarray | None = None
    mask: np.ndarray | None = None
    mask_visible: np.ndarray | None = None


class Camera(Entity, ABC):
    def __init__(self, name: str):
        Entity.__init__(self, name)

