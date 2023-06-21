import streamlit as st
from model.state import State

manual_text = """
1. Calibrate all cameras.
    For this, make sure all cameras are connected and recognized. Turn on the background monitor 
    and the robot. Make sure that the robot's pose can be read by this software and the monitor 
    can be controlled. Go to the Camera Calibration tab and move the robot so that the Charuco Board
    is visible in the live preview. Then, click on the Capture Image button. Repeat this process
    until you have captured enough images. Then, click on the Calibrate Intrinsics & Hand-Eye button.
    Confirm the successful calibration by checking the reprojection error and if the projected board
    is aligned with the real board in the live preview.
2. Register the pose of the object.
    
3. Acquire the Dataset.
"""
        
class Overview:
    def __init__(self, state: State) -> None:
        # TODO semantics editor: create an object, by selecting a mesh and a label
        # TODO show available cameras
        self._state = state
        st.title("Overview")
        st.info("here you can see a table with semantics: [obj_name, label, mesh]")
        st.info("a list of available cameras (for debugging)")

        st.subheader("Quick Manual")
        st.text(manual_text)
