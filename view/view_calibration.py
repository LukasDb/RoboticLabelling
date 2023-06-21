import streamlit as st
from model.state import State
from control.camera_calibration import CameraCalibration
import numpy as np


class ViewCalibration:
    def __init__(self, state: State) -> None:
        self._state = state
        st.title("1. Camera Calibration")
        c1, c2 = st.columns(2)

        selected_camera = c2.selectbox("Select Camera", ["Camera 1", "Camera 2"])

        c2.button("Capture Image")

        calibrator = CameraCalibration(selected_camera)

        selected = c2.radio(
            "Captured Images",
            calibrator.get_captured_images(),
            format_func=lambda x: f"Image {x:2}",
        )

        if c2.button("Calibrate Intrinsics & Hand-Eye"):
            calibrator.calibrate()

        live = calibrator.get_live_img()
        selected = calibrator.get_selected_img(selected)

        c1.image(
            live,
            caption="Live preview",
        )
        c1.image(
            selected,
            caption="Selected Image",
        )
