from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import numpy as np
from enum import Enum, auto
import numpy.typing as npt


import robolabel as rl
from robolabel import Event, Entity


@dataclass
class CamFrame:
    rgb: npt.NDArray[np.float64] | None = None
    rgb_R: npt.NDArray[np.float64] | None = None
    depth: npt.NDArray[np.float64] | None = None
    depth_R: npt.NDArray[np.float64] | None = None


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


class Camera(Entity, rl.Observable, ABC):
    width: int
    height: int

    def __init__(self, name: str):
        Entity.__init__(self, name)
        rl.Observable.__init__(self)

        self.robot: rl.robot.Robot | None = None
        import numpy.typing

        self._extrinsics: numpy.typing.NDArray[np.float64] | None = None
        self._intrinsics: npt.NDArray[np.float64] | None = None
        self._dist_coeffs: npt.NDArray[np.float64] | None = None

    @property
    async def pose(self) -> npt.NDArray[np.float64]:
        if self.robot is None or self._extrinsics is None:
            pose = self._pose
        else:
            pose = await self.robot.pose @ self._extrinsics
        return pose

    async def attach(self, robot: rl.robot.Robot, link_matrix: npt.NDArray[np.float64]):
        self.robot = robot
        self._extrinsics = link_matrix
        robot_pose = await self.robot.pose
        self._pose = robot_pose @ self._extrinsics
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
    def dist_coefficients(self):
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
        intrinsic_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        extrinsic_matrix: npt.NDArray[np.float64],
    ):
        self._intrinsics = intrinsic_matrix
        self._dist_coeffs = dist_coeffs
        self._extrinsics = extrinsic_matrix

        self.notify(Event.CAMERA_CALIBRATED)

    @abstractmethod
    def get_frame(self, depth_quality: DepthQuality) -> CamFrame:
        pass
