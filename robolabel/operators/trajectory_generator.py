from typing import List
import logging
from robolabel.labelled_object import LabelledObject
import numpy as np


class TrajectoryGenerator:
    def __init__(self, n_steps=20):
        self._n_steps = n_steps

        self._current_trajectory = None

    def generate_trajectory(self, active_objects: List[LabelledObject]):
        """generates a trajectory based on the selected objects"""
        if len(active_objects) == 0:
            logging.warning("No objects selected, cannot generate trajectory")
            return
        object_positions = np.array([o.get_position() for o in active_objects])
        center = np.mean(object_positions, axis=0)

    def get_current_trajectory(self) -> None | List[np.ndarray]:
        if self._current_trajectory is None:
            logging.warning("No trajectory generated yet")
            return None
        return self._current_trajectory
