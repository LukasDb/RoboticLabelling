import streamlit as st
from model.scene import Scene
from control.camera_calibration import CameraCalibrator
import numpy as np


class ViewCalibration:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        st.title("1. Camera Calibration")
        c1, c2 = st.columns(2)

        selected_camera = c2.selectbox("Select Camera", ["Camera 1", "Camera 2"])
        calibrator = CameraCalibrator(selected_camera)

        img_control = c2.columns(2)
        if img_control[0].button("Capture Image"):
            calibrator.capture_image()
        if img_control[1].button("Delete Captured Images"):
            calibrator.clear_cache()

        sel_img_index = c2.radio(
            "Captured Images",
            list(range(len(calibrator._captured_images))),
            format_func=lambda x: f"Image {x:2}",
        )

        if c2.button("Calibrate Intrinsics & Hand-Eye"):
            calibrator.calibrate()

        live = calibrator.get_live_img()
        sel_img = calibrator.get_selected_img(sel_img_index)

        c1.image(
            live,
            caption="Live preview",
        )
        if sel_img is not None:
            c1.image(
                sel_img,
                caption="Selected Image",
            )
