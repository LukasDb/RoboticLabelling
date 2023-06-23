import numpy as np
import streamlit as st
import cv2


class CameraCalibrator:
    def __init__(self, selected_camera):
        # needs to be persistent (which is not the case here)
        super().__init__()
        self.captured_images = []

    def setup(self):
        # TODO
        # - reset lighting
        # - create charuco board
        # - set background to charuco board
        pass

    def capture_image(self):
        # TODO actually capture image from camera
        # TODO also capture robot pose
        mock_img = np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)
        cv2.putText(
            mock_img,
            f"Captured Image {len(self.captured_images)}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        self.captured_images.append(mock_img)

    def calibrate(self):
        # TODO use robot poses and captured images to calibrate camera intrinsics and hand-eye
        #   similar to 6IMPOSE_Grasping camera calibration
        # TODO inform user about calibration result
        # TODO save calibration
        pass

    def get_live_img(self) -> np.ndarray:
        # return live image from camera with projected charuco board (if calibrated)
        # TODO get live img from camera
        # If calibrated: draw charuco board on image
        return np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)

    def get_selected_img(self, index) -> np.ndarray | None:
        # return selected image from camera with projected charuco board from cv2 detection
        if index is None:
            return None
        # project cv2 charuco board detection and estimated pose on image
        return self.captured_images[index]
