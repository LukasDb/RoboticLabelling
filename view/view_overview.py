import tkinter as tk
from tkinter import ttk
from model.scene import Scene

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


class Overview(tk.Frame):
    def __init__(self, parent, scene: Scene) -> None:
        # TODO semantics editor: create an object, by selecting a mesh and a label
        # TODO show available cameras
        tk.Frame.__init__(self, parent)
        self.scene = scene

        self.title = ttk.Label(self, text="Overview")
        self.title.grid()

        self.manual = tk.Label(self, text=manual_text)
        self.manual.grid()

        return