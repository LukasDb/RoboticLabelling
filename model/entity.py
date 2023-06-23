import numpy as np
from scipy.spatial.transform import Rotation as R


class Entity:
    """An entity is an object in the world that has a position and orientation"""

    def __init__(self, name: str):
        self.name = name
        self._pose = np.eye(4)  # global world 2 entity pose
        self._parent: Entity = None
        self._link_matrix: np.ndarray = None

    def set_position(self, position: np.ndarray):
        self.pose[:3, 3] = position

    def set_orientation(self, orientation: R):
        self.pose[:3, :3] = orientation.as_matrix()

    def get_position(self) -> np.ndarray:
        return self.pose[:3, 3]

    def get_orientation(self) -> R:
        return R.from_matrix(self.pose[:3, :3])

    @property
    def pose(self) -> np.ndarray:
        if self._parent is None:
            return self._pose
        else:
            return self._parent.pose @ self._link_matrix

    @pose.setter
    def pose(self, pose: np.ndarray):
        self._pose = pose

    def attach(self, parent: "Entity", link_matrix: np.ndarray):
        self._parent = parent
        self._link_matrix = link_matrix
        self._pose = self._parent.pose @ self._link_matrix
