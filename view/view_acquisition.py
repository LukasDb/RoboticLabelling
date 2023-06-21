from model.state import State
import streamlit as st


class ViewAcquisition:
    def __init__(self, state: State) -> None:
        self._state = state

        st.title("3. Acquisition")
