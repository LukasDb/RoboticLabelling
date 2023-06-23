import tkinter as tk
from tkinter import ttk
from model.scene import Scene
from .view_overview import Overview
from .view_calibration import ViewCalibration
from .view_pose_registration import ViewPoseRegistration
from .view_acquisition import ViewAcquisition


class App:
    OVERVIEW = "overview"
    CAMERA_CALIBRATION = "camera_calibration"
    POSE_REGISTRATION = "pose_registration"
    ACQUISITION = "acquisition"

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Robotic Labelling")

        self.scene = Scene()
        # add cameras, robots and link to scene

    def run(self):
        # c1, c2, c3, c4 = st.columns(4)

        tabs = ttk.Notebook(self.root)
        tabs.pack(expand=True, fill=tk.BOTH)

        ov = Overview(tabs, self.scene)
        tabs.add(ov, text="Overview")

        cal = ViewCalibration(tabs, self.scene)
        tabs.add(cal, text="1. Camera Calibration")

        reg = ViewPoseRegistration(tabs, self.scene)
        tabs.add(reg, text="2. Pose Registration")

        acq = ViewAcquisition(tabs, self.scene)
        tabs.add(acq, text="3. Acquisition")

        self.root.mainloop()
        return

        if c1.button("Overview", use_container_width=True):
            st.session_state.tab = self.OVERVIEW
        if c2.button("1\. Camera Calibration", use_container_width=True):
            st.session_state.tab = self.CAMERA_CALIBRATION
        if c3.button("2\. Pose Registration", use_container_width=True):
            st.session_state.tab = self.POSE_REGISTRATION
        if c4.button("3\. Acquisition", use_container_width=True):
            st.session_state.tab = self.ACQUISITION

        if st.session_state.tab == self.OVERVIEW:
            Overview(self.scene)
        elif st.session_state.tab == self.CAMERA_CALIBRATION:
            ViewCalibration(self.scene)
        elif st.session_state.tab == self.POSE_REGISTRATION:
            ViewPoseRegistration(self.scene)
        elif st.session_state.tab == self.ACQUISITION:
            ViewAcquisition(self.scene)
