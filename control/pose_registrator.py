import numpy as np
from scipy.spatial.transform import Rotation as R
from .image_cache import ImageCache
import streamlit as st
import cv2


class PoseRegistrator(ImageCache):
    # handles the initial object pose registration
    def __init__(self):
        ImageCache.__init__(self)
        # TODO set initial pose to the center of the charuco board
        # TODO set background monitor so some 'easy' background
        # TODO set lighting to standard lighting
        if "register_pos" not in st.session_state:
            self.reset()

        pass

    def capture_image(self):
        # TODO actually capture image from camera
        # TODO also capture robot pose
        mock_img = np.random.uniform(size=(480, 640, 3), high=255).astype(np.uint8)
        cv2.putText(
            mock_img,
            f"Captured Image {len(self._captured_images)}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        self._captured_images.append(mock_img)

    def get_position(self):
        return st.session_state.register_pos

    def set_position(self, position):
        st.session_state.register_pos = position

    def get_orientation(self):
        return st.session_state.register_rot

    def set_orientation(self, orientation):
        st.session_state.register_rot = orientation

    def reset(self):
        st.session_state.register_pos = [0.0, 0.0, 0.0]
        st.session_state.register_rot = R.from_quat([0, 0, 0, 1])

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
        return self._captured_images[index]
