import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import numpy as np
import json
import logging
from pathlib import Path
import asyncio

from robolabel.scene import Scene
from robolabel.observer import Event, Observable, Observer
from robolabel.robot import MockRobot, FanucCRX10iAL
from robolabel.camera import DemoCam, Realsense, ZedCamera
from robolabel.labelled_object import LabelledObject
from robolabel.operators import CameraCalibrator, PoseRegistrator
from robolabel.gui import Overview, ViewAcquisition, ViewPoseRegistration, ViewCalibration


class App:
    OVERVIEW = "overview"
    CAMERA_CALIBRATION = "camera_calibration"
    POSE_REGISTRATION = "pose_registration"
    ACQUISITION = "acquisition"

    def __init__(self) -> None:
        self.root = tk.Tk()
        # self.tk_updater = loop.create_task(self._tk_updater(), name="TK Updater")
        self.stop = False
        self.root.title("Robotic Labelling")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.scene = Scene()
        # --- load controllers ---
        self.calibrator = CameraCalibrator(self.scene)
        self.pose_registrator = PoseRegistrator(self.scene)

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

        for cam in ZedCamera.get_available_devices():
            self.scene.add_camera(cam)

        state_path = Path("app_state.json")
        if state_path.exists():
            with state_path.open("r") as f:
                data = json.load(f)
            self.load_state(data)

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

        home_robot_menu = tk.Menu(self.menubar, tearoff=0)
        for name, robot in self.scene.robots.items():
            home_robot_menu.add_command(
                label=f"Set {name} home pose",
                command=lambda robot=robot: robot.set_current_as_homepose(),
            )
        self.menubar.add_cascade(label="Home Robot", menu=home_robot_menu)

        self.root.config(menu=self.menubar)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.overview = Overview(self.root, self.scene, self.calibrator, self.pose_registrator)
        tabs = ttk.Notebook(self.root)
        self.overview.grid(sticky=tk.NSEW, pady=10)
        tabs.grid(sticky=tk.NSEW, pady=10)

        self.cal = ViewCalibration(tabs, self.scene, self.calibrator)
        tabs.add(self.cal, text="1. Camera Calibration")

        reg = ViewPoseRegistration(tabs, self.scene, self.pose_registrator)
        tabs.add(reg, text="2. Pose Registration")

        acq = ViewAcquisition(tabs, self.scene)
        tabs.add(acq, text="3. Acquisition")

    def _on_close(self):
        logging.warning("Closing app...")
        state_path = Path("app_state.json")
        self.save_state(state_path)
        self.stop = True

    def run(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._tk_updater())
        loop.close()

    async def _tk_updater(self):
        while not self.stop:
            try:
                self.root.update()
                await asyncio.sleep(1 / 30)
            except Exception:
                break

        await asyncio.sleep(1.0)
        logging.info("Tk loop finished.")

    def load_state(self, data):
        self.calibrator.load(data["camera_calibration"])
        for obj_data in data["objects"]:
            obj = LabelledObject(
                obj_data["name"],
                Path(obj_data["mesh_path"]),
                obj_data["semantic_color"],
            )

            if obj_data["registered"]:
                obj.register_pose(np.array(obj_data["pose"]))

            self.scene.add_object(obj)

        for name, pose in data["robots"].items():
            if pose == "none":
                continue
            self.scene.robots[name].home_pose = np.array(pose)

    def save_state(self, file: Path):
        data = {
            "camera_calibration": self.calibrator.dump(),
            "objects": [
                {
                    "name": name,
                    "mesh_path": str(obj.mesh_path),
                    "pose": obj.pose.tolist(),
                    "registered": obj.registered,
                    "semantic_color": obj.semantic_color,
                }
                for name, obj in self.scene.objects.items()
            ],
            "robots": {
                name: "none" if robot.home_pose is None else robot.home_pose.tolist()
                for name, robot in self.scene.robots.items()
            },
        }

        with Path(file.name).open("w") as f:
            json.dump(data, f, indent=2)

    def _on_load_config(self):
        file = filedialog.askopenfile(
            title="Select Configuration file",
            filetypes=(("configuration files", "*.config"), ("all files", "*.*")),
        )
        with Path(file.name).open("r") as f:
            data = json.load(f)
        self.load_state(data)

    def _on_save_config(self) -> None:
        default_name = "scene"

        file = filedialog.asksaveasfile(
            title="Save Configuration file",
            filetypes=(("configuration files", "*.config"), ("all files", "*.*")),
            defaultextension=".config",
            initialfile=default_name,
        )
        self.save_state(file)

    def _on_add_object(self) -> None:
        file = filedialog.askopenfile(
            title="Select Object Ply file",
            filetypes=(("ply files", "*.ply"), ("all files", "*.*")),
        )

        if file is None:
            return

        path = Path(file.name)

        obj_name = path.stem
        if obj_name in self.scene.objects:
            i = 1
            while f"{obj_name}_{i}" in self.scene.objects:
                i += 1
            obj_name = f"{obj_name}_{i}"
        new_object = LabelledObject(obj_name, path)
        monitor_pose = self.scene.background.pose
        new_object.pose = monitor_pose  # add this as inital position
        self.scene.add_object(new_object)
