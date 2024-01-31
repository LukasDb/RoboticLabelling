import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


class BaseEntity:
    def __init__(self, name: str):
        self.name = name
        self._pose = np.eye(4)  # global world 2 entity pose

    def __str__(self) -> str:
        return f"{self.name}"


class Entity(BaseEntity):
    """An entity is an object in the world that has a position and orientation"""

    def get_position(self) -> npt.NDArray[np.float64]:
        pose = self.get_pose()
        return pose[:3, 3]

    def get_orientation(self) -> R:
        pose = self.get_pose()
        return R.from_matrix(pose[:3, :3])

    def get_pose(self) -> npt.NDArray[np.float64]:
        return self._pose

    def set_pose(self, pose: npt.NDArray[np.float64]) -> None:
        self._pose = pose


class AsyncEntity(BaseEntity):
    """An entity is an object in the world that has a position and orientation"""

    async def get_position(self) -> npt.NDArray[np.float64]:
        pose = await self.get_pose()
        return pose[:3, 3]

    async def get_orientation(self) -> R:
        pose = await self.get_pose()
        return R.from_matrix(pose[:3, :3])

    async def get_pose(self) -> npt.NDArray[np.float64]:
        return self._pose

    def set_pose(self, pose: npt.NDArray[np.float64]) -> None:
        self._pose = pose
