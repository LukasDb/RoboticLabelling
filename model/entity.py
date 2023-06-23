import numpy as np
from scipy.spatial.transform import Rotation as R


class Entity:
    """An entity is an object in the world that has a position and orientation"""

    def __init__(self, name: str):
        self.name = name
        self._position = np.zeros((3,))
        self._orientation = R.from_quat([0, 0, 0, 1])
        self.parent: Entity = None
        self.link_matrix: np.ndarray = None

    def set_position(self, position: np.ndarray):
        self._position = position

    def set_orientation(self, orientation: R):
        self._orientation = orientation

    def get_position(self) -> np.ndarray:
        return self._position

    def get_orientation(self) -> R:
        return self._orientation

    def get_transform(self) -> np.ndarray:
        # return 4x4 transformation matrix
        return np.block(
            [
                [self.get_orientation().as_matrix(), self.get_position()[:, None]],
                [np.zeros((1, 3)), 1],
            ]
        )

    def attach(self, parent: "Entity", link_matrix: np.ndarray):
        self.parent = parent
        self.link_matrix = link_matrix
