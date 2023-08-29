import numpy as np
from scipy.spatial.transform import Rotation as R
from threading import Lock


def threadsafe(func):
    def wrapper(self: "Entity", *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class Entity:
    """An entity is an object in the world that has a position and orientation"""

    def __init__(self, name: str):
        self.lock = Lock()
        self.name = name
        self._pose = np.eye(4)  # global world 2 entity pose
        self.parent: Entity = None
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
    @threadsafe
    def pose(self) -> np.ndarray:
        if self.parent is None:
            pose = self._pose
        else:
            pose = self.parent.pose @ self._link_matrix
        return pose

    @pose.setter
    @threadsafe
    def pose(self, pose: np.ndarray):
        self._pose = pose

    @threadsafe
    def attach(self, parent: "Entity", link_matrix: np.ndarray):
        self.parent = parent
        self._link_matrix = link_matrix
        self._pose = self.parent.pose @ self._link_matrix

    @threadsafe
    def detach(self):
        self.parent = None
        self._link_matrix = None

    def __str__(self) -> str:
        return f"{self.name}"
