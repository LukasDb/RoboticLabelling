import numpy as np


class CameraCalibration:
    def __init__(self, selected_camera):
        # needs to be persistent (which is not the case here)
        self._captured_images =[]  

        for i in range(4):
            self._captured_images.append(
                np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)
            )

    def setup(self):
        # TODO
        # - reset lighting
        # - create charuco board
        # - set background to charuco board
        pass

    def calibrate(self):
        pass

    def get_live_img(self) -> np.ndarray:
        return np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)

    def get_selected_img(self, index) -> np.ndarray:
        return self._captured_images[index]

    def get_captured_images(self):
        return [i for i in range(len(self._captured_images))]
