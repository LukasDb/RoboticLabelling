from model.scene import Scene
import streamlit as st


class ViewAcquisition:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene

        st.title("3. Acquisition")
