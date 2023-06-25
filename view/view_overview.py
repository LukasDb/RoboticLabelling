import tkinter as tk
from tkinter import ttk
from model.scene import Scene
from model.camera.camera import Camera
from model.observer import Observer, Event
from typing import Dict
import numpy as np

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


class Overview(Observer, tk.Frame):
    def __init__(self, parent, scene: Scene) -> None:
        # TODO semantics editor: create an object, by selecting a mesh and a label
        # TODO show available cameras
        tk.Frame.__init__(self, parent)
        Observer.__init__(self)
        self._scene = scene

        for cam in self._scene.cameras.values():
            self.listen_to(cam)

        self.title = ttk.Label(self, text="Overview")

        self.robot_check = self.setup_robot_pose()
        self.bg_setup = self.setup_background_window()
        self._cam_rows: Dict[str, Dict] = {}
        self.cam_table = self.setup_camera_table()

        self.manual = tk.Label(self, text=manual_text)

        self.columnconfigure(0, weight=1)

        self.title.grid()
        self.robot_check.grid()
        self.bg_setup.grid()
        self.cam_table.grid(sticky=tk.NSEW)
        self.manual.grid()

    def update(self, subject, event: Event, *args, **kwargs):
        if event == Event.CAMERA_CALIBRATED:
            # update calibrated column
            self._update_cam_row(subject.unique_id)

    def setup_camera_table(self):
        self.cam_overview = tk.Frame(self)
        self.cam_overview.columnconfigure(0, weight=1)

        columns = ["Camera", "Calibrated?", "Attached to"]
        headings = tk.Frame(self.cam_overview)
        headings.grid(row=0, column=0, sticky=tk.EW)
        for i, c in enumerate(columns):
            headings.columnconfigure(i, weight=1)
            label = tk.Label(headings, text=c)
            label.grid(row=0, column=i, sticky=tk.EW)
        # add separator
        sep = ttk.Separator(self.cam_overview, orient=tk.HORIZONTAL)
        sep.grid(row=1, column=0, sticky=tk.EW)

        self.cam_table = tk.Frame(self.cam_overview)
        self.cam_table.columnconfigure(0, weight=1)
        self.cam_table.grid(row=2, column=0, sticky=tk.NSEW)

        for i, c in enumerate(self._scene.cameras.values()):
            row = tk.Frame(self.cam_table)
            row.grid(row=i, column=0, sticky=tk.EW)

            row.columnconfigure(0, weight=1)
            row.columnconfigure(1, weight=1)
            row.columnconfigure(2, weight=1)

            # add camera name
            cam_name = tk.Label(row)
            cam_name.grid(row=0, column=0, sticky=tk.EW)
            # add calibrated
            calibrated = tk.Label(row)
            calibrated.grid(row=0, column=1, sticky=tk.EW)
            # add parent
            # parent = tk.Label(row)
            parent = ttk.Combobox(row, values=["-"] + list(self._scene.robots.keys()))
            parent.set("-")
            parent.bind(
                "<<ComboboxSelected>>",
                lambda event, cam_unique_id=c.unique_id, parent=parent: self.on_attach_cam(
                    cam_unique_id, parent
                ),
            )
            parent.grid(row=0, column=2, sticky=tk.EW)

            self._cam_rows[c.unique_id] = {
                "cam_name": cam_name,
                "calibrated": calibrated,
                "parent": parent,
            }

            # update values
            self._update_cam_row(c.unique_id)

        return self.cam_overview

    def on_attach_cam(self, cam_unique_id, parent):
        if parent.get() == "-":
            self._scene.cameras[cam_unique_id].detach()
        else:
            robot = self._scene.robots[parent.get()]
            link_mat = self._scene.cameras[cam_unique_id].extrinsic_matrix
            link_mat = link_mat if link_mat is not None else np.eye(4)
            self._scene.cameras[cam_unique_id].attach(robot, link_mat)

    def _update_cam_row(self, cam_unique_id):
        cam = self._scene.cameras[cam_unique_id]
        row = self._cam_rows[cam_unique_id]
        row["cam_name"].configure(text=f"{cam.name} ({cam.unique_id})")
        row["calibrated"].configure(
            text="Yes" if cam.intrinsic_matrix is not None else "No"
        )
        row["parent"].set("-" if cam.parent is None else cam.parent.name)
        # row["parent"].configure(text="-" if cam.parent is None else cam.parent.name)

    def setup_background_window(self):
        # add button with a label to "Setup up background monitor"
        self.bg_frame = ttk.Frame(self)
        self.btn_setup_bg = ttk.Button(
            self.bg_frame,
            text="Setup Background Monitor",
            command=self._scene.background.setup_window,
        )
        self.bg_label = ttk.Label(self.bg_frame, text="Setup Background Monitor")

        self.btn_setup_bg.grid(row=0, column=0, padx=5)
        self.bg_label.grid(row=0, column=1, padx=5)
        return self.bg_frame

    def setup_robot_pose(self):
        self.robot_frame = ttk.Frame(self)
        self.btn_update_robot = ttk.Button(
            self.robot_frame, text="Update Robot", command=self.update_robot
        )
        self.pose_label = ttk.Label(self.robot_frame, text="Pose: ")

        self.btn_update_robot.grid(row=0, column=0, padx=5)
        self.pose_label.grid(row=0, column=1, padx=5)

        return self.robot_frame

    def update_robot(self):
        robot = list(self._scene.robots.values())[0]
        position = robot.get_position()
        orientation = robot.get_orientation().as_euler("xyz", degrees=True)
        # update pose label
        self.pose_label["text"] = f"Pose: {position}, {orientation}"
