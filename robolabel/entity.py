import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


class Entity:
    """An entity is an object in the world that has a position and orientation"""

    def __init__(self, name: str):
        self.name = name
        self._pose = np.eye(4)  # global world 2 entity pose

    async def get_position(self) -> npt.NDArray[np.float64]:
        pose = await self.pose
        return pose[:3, 3]

    async def get_orientation(self) -> R:
        pose = await self.pose
        return R.from_matrix(pose[:3, :3])

    @property
    async def pose(self) -> npt.NDArray[np.float64]:
        return self._pose

    @pose.setter
    def pose(self, pose: npt.NDArray[np.float64]):
        self._pose = pose

    def __str__(self) -> str:
        return f"{self.name}"
