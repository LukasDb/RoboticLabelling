import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import numpy as np
import pickle
from pathlib import Path

from model.scene import Scene
from model.fanuc_crx10ial import FanucCRX10iAL
from model.demo_robot import MockRobot
from model.camera.realsense import Realsense
#from model.camera.zed import ZedCamera
from model.camera.demo_cam import DemoCam
from model.labelled_object import LabelledObject

from control.camera_calibration import CameraCalibrator
from control.pose_registrator import PoseRegistrator

from .view_overview import Overview
from .view_calibration import ViewCalibration
from .view_pose_registration import ViewPoseRegistration
from .view_acquisition import ViewAcquisition

import logging
import coloredlogs

coloredlogs.install()
logging.basicConfig(level=logging.DEBUG)


class App:
    OVERVIEW = "overview"
    CAMERA_CALIBRATION = "camera_calibration"
    POSE_REGISTRATION = "pose_registration"
    ACQUISITION = "acquisition"

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Robotic Labelling")
        self.scene = Scene()
        # --- load controllers ---
        self.calibrator = CameraCalibrator(self.scene)
        self.pose_registrator = PoseRegistrator(self.scene)

        ## ---  build GUI ---
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        # add save menu
        self.filemenu.add_command(label="Save...", command=self._on_save_config)
        self.filemenu.add_command(label="Load...", command=self._on_load_config)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.objectmenu = tk.Menu(self.menubar, tearoff=0)
        self.objectmenu.add_command(label="Add Object", command=self._on_add_object)
        self.menubar.add_cascade(label="Object", menu=self.objectmenu)

        self.root.config(menu=self.menubar)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.overview = Overview(
            self.root, self.scene, self.calibrator, self.pose_registrator
        )
        tabs = ttk.Notebook(self.root)
        self.overview.grid(sticky=tk.NSEW, pady=10)
        tabs.grid(sticky=tk.NSEW, pady=10)

        self.cal = ViewCalibration(tabs, self.scene, self.calibrator)
        tabs.add(self.cal, text="1. Camera Calibration")

        reg = ViewPoseRegistration(tabs, self.scene, self.pose_registrator)
        tabs.add(reg, text="2. Pose Registration")

        acq = ViewAcquisition(tabs, self.scene)
        tabs.add(acq, text="3. Acquisition")

        ## --- Assemble scene ---
        # for testing
        mock_robot = MockRobot()
        self.scene.add_robot(mock_robot)
        self.scene.add_camera(DemoCam("Demo Cam"))

        crx = FanucCRX10iAL()
        self.scene.add_robot(crx)

        # find connected realsense devices
        for cam in Realsense.get_available_devices():
            self.scene.add_camera(cam)

        #for cam in ZedCamera.get_available_devices():
        #    self.scene.add_camera(cam)

    def run(self):
        tk.mainloop()

    def _on_load_config(self):
        file = filedialog.askopenfile(
            title="Select Configuration file",
            filetypes=(("configuration files", "*.config"), ("all files", "*.*")),
        )
        with Path(file.name).open("rb") as f:
            data = pickle.load(f)

        self.calibrator.load(data["camera_calibration"])

        for obj_data in data["objects"]:
            obj = LabelledObject(
                obj_data["name"], obj_data["mesh_path"], obj_data["semantic_color"]
            )
            if obj_data["registered"]:
                obj.register_pose(obj_data["pose"])
            self.scene.add_object(obj)

    def _on_save_config(self):
        default_name = "scene"

        file = filedialog.asksaveasfile(
            title="Save Configuration file",
            filetypes=(("configuration files", "*.config"), ("all files", "*.*")),
            defaultextension=".config",
            initialfile=default_name,
        )
        data = {
            "camera_calibration": self.calibrator.dump(),
            "objects": [
                {
                    "name": name,
                    "mesh_path": obj.mesh_path,
                    "pose": obj.pose,
                    "registered": obj.registered,
                    "semantic_color": obj.semantic_color,
                }
                for name, obj in self.scene.objects.items()
            ],
        }

        with Path(file.name).open("wb") as f:
            pickle.dump(data, f)

    def _on_add_object(self):
        file = filedialog.askopenfile(
            title="Select Object Ply file",
            filetypes=(("ply files", "*.ply"), ("all files", "*.*")),
        )

        if file is None:
            return

        path = Path(file.name)

        new_object = LabelledObject(path.stem, path)
        monitor_pose = self.scene.background.pose
        new_object.pose = monitor_pose  # add this as inital position
        self.scene.add_object(new_object)
