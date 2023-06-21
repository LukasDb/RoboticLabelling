import numpy as np
from scipy.spatial.transform import Rotation as R


class Entity:
    """An entity is an object in the world that has a position and orientation"""

    def __init__(self):
        self._position = np.zeros((3,))
        self._orientation = R.from_quat([0, 0, 0, 1])

    def set_position(self, position: np.ndarray):
        self._position = position

    def set_orientation(self, orientation: R):
        self._orientation = orientation

    def get_transform(self)->np.ndarray:
        # return 4x4 transformation matrix
        return np.block(
            [
                [self._orientation.as_matrix(), self._position[:, None]],
                [np.zeros((1, 3)), 1],
            ]
        )
