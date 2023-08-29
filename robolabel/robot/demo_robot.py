import numpy as np
from robolabel.robot.robot import Robot


class MockRobot(Robot):
    def __init__(self):
        super().__init__(name="mock")

    def move_to(self, pose: np.ndarray, block=True):
        raise NotImplementedError
