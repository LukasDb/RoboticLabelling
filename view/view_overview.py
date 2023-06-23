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

        self.robot_check = self.robot_check_widget()

        self.manual = tk.Label(self, text=manual_text)

        self.title.grid()
        self.robot_check.grid()
        self.manual.grid()

    def robot_check_widget(self):
        self.robot_frame = ttk.Frame(self)
        self.btn_update_robot = ttk.Button(
            self.robot_frame, text="Update Robot", command=self.update_robot
        )
        self.pose_label = ttk.Label(self.robot_frame, text="Pose: ")

        self.btn_update_robot.grid(row=0, column=0, padx=5)
        self.pose_label.grid(row=0, column=1, padx=5)

        return self.robot_frame

    def update_robot(self):
        robot = self.scene.robots[0]
        position = robot.get_position()
        orientation = robot.get_orientation().as_euler("xyz", degrees=True)
        # update pose label
        self.pose_label["text"] = f"Pose: {position}, {orientation}"
