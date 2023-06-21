import streamlit as st
from model.state import State


class ViewPoseRegistration:
    def __init__(self, state: State) -> None:
        self._state = state
        st.title("2. Pose Registration")
