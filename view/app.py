import streamlit as st
from model.state import State
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
        st.set_page_config(layout="wide")
        self._state = State()
        if "tab" not in st.session_state:
            st.session_state.tab = self.OVERVIEW

    def run(self):
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Overview", use_container_width=True):
            st.session_state.tab = self.OVERVIEW
        if c2.button("1\. Camera Calibration", use_container_width=True):
            st.session_state.tab = self.CAMERA_CALIBRATION
        if c3.button("2\. Pose Registration", use_container_width=True):
            st.session_state.tab = self.POSE_REGISTRATION
        if c4.button("3\. Acquisition", use_container_width=True):
            st.session_state.tab = self.ACQUISITION

        if st.session_state.tab == self.OVERVIEW:
            Overview(self._state)
        elif st.session_state.tab == self.CAMERA_CALIBRATION:
            ViewCalibration(self._state)
        elif st.session_state.tab == self.POSE_REGISTRATION:
            ViewPoseRegistration(self._state)
        elif st.session_state.tab == self.ACQUISITION:
            ViewAcquisition(self._state)
