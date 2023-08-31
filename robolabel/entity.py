import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


class Entity:
    """An entity is an object in the world that has a position and orientation"""

    def __init__(self, name: str):
        self.name = name
        self._pose: npt.NDArray[np.float64] = np.eye(4)  # global world 2 entity pose

    def set_position(self, position: npt.NDArray[np.float64]):
        self.pose[:3, 3] = position

    def set_orientation(self, orientation: R) -> None:
        self.pose[:3, :3] = orientation.as_matrix()

    def get_position(self) -> npt.NDArray[np.float64]:
        return self.pose[:3, 3]

    def get_orientation(self) -> R:
        return R.from_matrix(self.pose[:3, :3])

    @property
    def pose(self) -> npt.NDArray[np.float64]:
        return self._pose

    @pose.setter
    def pose(self, pose: npt.NDArray[np.float64]):
        self._pose = pose

    def __str__(self) -> str:
        return f"{self.name}"
