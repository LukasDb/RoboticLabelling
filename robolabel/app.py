import tkinter as tk
from tkinter import ttk
import time
import tkinter.filedialog as filedialog
import numpy as np
import json
import logging
from pathlib import Path
import asyncio

import robolabel as rl


class App:
    OVERVIEW = "overview"
    CAMERA_CALIBRATION = "camera_calibration"
    POSE_REGISTRATION = "pose_registration"
    ACQUISITION = "acquisition"

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.stop = False
        self.root.title("Robotic Labelling")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.scene = rl.Scene()

        self.assemble_gui()
        self.populate_scene()

        self.t_last = time.perf_counter()
        self.FPS = 30

    def populate_scene(self):
        ## --- Assemble scene ---
        # for testing
        mock_robot = rl.robot.MockRobot()
        self.scene.add_robot(mock_robot)
        self.scene.add_camera(rl.camera.DemoCam("Demo Cam"))

        crx = rl.robot.FanucCRX10iAL()
        self.scene.add_robot(crx)

        # find connected realsense devices
        for cam in rl.camera.Realsense.get_available_devices():
            self.scene.add_camera(cam)

        #for cam in rl.camera.ZedCamera.get_available_devices():
        #    self.scene.add_camera(cam)

    def assemble_gui(self):
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        # add save menu
        self.filemenu.add_command(label="Save...", command=self._on_save_config)
        self.filemenu.add_command(label="Load...", command=self._on_load_config)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self._on_close)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.objectmenu = tk.Menu(self.menubar, tearoff=0)
        self.objectmenu.add_command(label="Add Object", command=self._on_add_object)
        self.menubar.add_cascade(label="Object", menu=self.objectmenu)

        # --- load controllers ---
        self.calibrator = rl.operators.CameraCalibrator(self.scene)
        self.pose_registration = rl.operators.PoseRegistration(self.scene)

        # home_robot_menu = tk.Menu(self.menubar, tearoff=0)
        # for name, robot in self.scene.robots.items():
        #     home_robot_menu.add_command(
        #         label=f"Set {name} home pose",
        #         command=lambda robot=robot: self._on_set_home(robot),
        #     )
        # self.menubar.add_cascade(label="Home Robot", menu=home_robot_menu)

        self.root.config(menu=self.menubar)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.overview = rl.gui.Overview(
            self.root, self.scene, self.calibrator, self.pose_registration
        )
        self.tabs = ttk.Notebook(self.root)
        self.overview.grid(sticky=tk.NSEW, pady=10)
        self.tabs.grid(sticky=tk.NSEW, pady=10)

        self.cal = rl.gui.ViewCalibration(self.tabs, self.scene, self.calibrator)
        self.tabs.add(self.cal, text="1. Camera Calibration")

        reg = rl.gui.ViewPoseRegistration(self.tabs, self.scene, self.pose_registration)
        self.tabs.add(reg, text="2. Pose Registration")

        self.acquisition = rl.gui.ViewAcquisition(self.tabs, self.scene)
        self.tabs.add(self.acquisition, text="3. Acquisition")

        # register callback on tab change
        self.tabs.bind(sequence="<<NotebookTabChanged>>", func=lambda _: self._on_tab_change())

    def _on_tab_change(self):
        open_tab = self.tabs.select()
        logging.debug(f"Tab changed to {open_tab}")
        if "viewposeregistration" in open_tab:
            self.scene.change_mode("registration")
        elif "viewcalibration" in open_tab:
            self.scene.change_mode("calibration")
        elif "viewacquisition" in open_tab:
            self.scene.change_mode("acquisition")

    def _on_close(self):
        logging.warning("Closing app...")
        self.stop = True

    def run(self):
        loop = asyncio.get_event_loop()
        state_path = Path("app_state.config")
        self.load_state(state_path)

        try:
            loop.run_until_complete(self._tk_updater())
        except KeyboardInterrupt:
            pass

        state_path = Path("app_state.config")
        self.save_state(state_path)

    async def _tk_updater(self):
        while not self.stop:
            try:
                self.root.update()
                await self.overview.update_live_preview()
                await asyncio.sleep(1 / 30)
            except Exception:
                break
        logging.info("Tk loop finished.")

    @rl.as_async_task
    async def load_state(self, file: Path) -> None:
        try:
            with Path(file).open("r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"Could not load state from {file}")
            return

        await self.calibrator.load(data["camera_calibration"])
        for obj_data in data["objects"]:
            obj = rl.LabelledObject(
                obj_data["name"],
                Path(obj_data["mesh_path"]),
                obj_data["semantic_color"],
            )

            obj.pose = np.array(obj_data["pose"])

            self.scene.add_object(obj)

        for name, pose in data["robots"].items():
            if pose == "none":
                continue
            self.scene.robots[name].home_pose = np.array(pose)

    @rl.as_async_task
    async def save_state(self, filepath: Path) -> None:
        data = {
            "camera_calibration": await self.calibrator.dump(),
            "objects": [
                {
                    "name": name,
                    "mesh_path": str(obj.mesh_path),
                    "pose": (await obj.pose).tolist(),
                    "semantic_color": obj.semantic_color,
                }
                for name, obj in self.scene.objects.items()
            ],
            "robots": {
                name: "none" if robot.home_pose is None else robot.home_pose.tolist()
                for name, robot in self.scene.robots.items()
            },
        }
        # with Path(file.name).open("w") as f:
        with filepath.open("w") as F:
            json.dump(data, F, indent=2)

    def _on_load_config(self):
        file = filedialog.askopenfilename(
            title="Select Configuration file",
            filetypes=(("configuration files", "*.config"), ("all files", "*.*")),
        )
        self.load_state(Path(file))

    def _on_save_config(self) -> None:
        default_name = "scene"

        file_name = filedialog.asksaveasfilename(
            title="Save Configuration file",
            filetypes=(("configuration files", "*.config"), ("all files", "*.*")),
            defaultextension=".config",
            initialfile=default_name,
        )
        try:
            filepath = Path(file_name).resolve()
        except Exception as e:
            logging.error("Could not save file: %s", e)
            return

        self.save_state(filepath)

    @rl.as_async_task
    async def _on_add_object(self) -> None:
        file = filedialog.askopenfilename(
            title="Select Object Ply file",
            filetypes=(("ply files", "*.ply"), ("all files", "*.*")),
        )

        if file is None:
            return

        path = Path(file)

        obj_name = path.stem
        if obj_name in self.scene.objects:
            i = 1
            while f"{obj_name}.{i:03}" in self.scene.objects:
                i += 1
            obj_name = f"{obj_name}.{i:03}"
        new_object = rl.LabelledObject(obj_name, path)
        monitor_pose = await self.scene.background.pose
        new_object.pose = monitor_pose  # add this as initial position
        self.scene.add_object(new_object)
