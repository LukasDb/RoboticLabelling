import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


class PoseRegistrator:
    # handles the initial object pose registration
    def __init__(self):
        super().__init__()
        # TODO set initial pose to the center of the charuco board
        # TODO set background monitor so some 'easy' background
        # TODO set lighting to standard lighting
        self.captured_images = []
        self.registered_position = None
        self.registered_orientation = None
        self.reset()

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

    def get_position(self):
        return self.registered_position

    def set_position(self, position):
        self.registered_position = position

    def get_orientation(self):
        return self.registered_orientation

    def set_orientation(self, orientation):
        self.registered_orientation = orientation

    def reset(self):
        self.registered_position = [0.0, 0.0, 0.0]
        self.registered_orientation = R.from_quat([0, 0, 0, 1])

    def optimize_pose(self):
        # TODO optimize pose in each image using ICP
        # TODO optimize object pose over all images
        # TODO inform user about optimization result
        # TODO save optimized pose to the scene
        pass

    def get_live_img(self):
        # TODO draw object at globally optimized pose on image
        return np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)

    def get_selected_img(self, index):
        # TODO draw object pose on image (for each ICP optimized pose)
        if index is None:
            return None
        return self.captured_images[index]
