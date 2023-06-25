import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import ttk
from model.scene import Scene
import numpy as np

from model.fanuc_crx10ial import FanucCRX10iAL
from model.camera.realsense import Realsense
from model.camera.demo_cam import DemoCam


from .view_overview import Overview
from .view_calibration import ViewCalibration
from .view_pose_registration import ViewPoseRegistration
from .view_acquisition import ViewAcquisition


class App:
    OVERVIEW = "overview"
    CAMERA_CALIBRATION = "camera_calibration"
    POSE_REGISTRATION = "pose_registration"
    ACQUISITION = "acquisition"

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Robotic Labelling")

        self.scene = Scene()

        crx = FanucCRX10iAL()
        self.scene.add_robot(crx)

        # find connected realsense devices
        for cam in Realsense.get_available_devices():
            self.scene.add_camera(cam)
            cam.attach(crx, np.eye(4))

        # add demo cameras
        self.scene.add_camera(DemoCam("Demo Cam"))
        c1 = DemoCam("demo_cam_1")
        c1.attach(crx, np.eye(4))
        self.scene.add_camera(c1)
        self.scene.add_camera(DemoCam("demo_cam_2"))
        self.scene.add_camera(DemoCam("demo_cam_3"))
        self.scene.add_camera(DemoCam("demo_cam_4"))
        self.scene.add_camera(DemoCam("demo_cam_5"))
        self.scene.add_camera(DemoCam("demo_cam_6"))
        self.scene.add_camera(DemoCam("demo_cam_7"))

    def run(self):
        print("Running...")
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(
            label="Load Calibration...", command=self.on_menu_load_calib
        )
        self.filemenu.add_command(
            label="Save Calibration as...", command=self.on_menu_save_calib
        )
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.root.config(menu=self.menubar)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        tabs = ttk.Notebook(self.root)
        tabs.grid(padx=10, pady=10, sticky=tk.NSEW)

        self.overview = Overview(tabs, self.scene)
        tabs.add(self.overview, text="Overview")

        self.cal = ViewCalibration(tabs, self.scene)
        tabs.add(self.cal, text="1. Camera Calibration")

        reg = ViewPoseRegistration(tabs, self.scene)
        tabs.add(reg, text="2. Pose Registration")

        acq = ViewAcquisition(tabs, self.scene)
        tabs.add(acq, text="3. Acquisition")

        tk.mainloop()

    def on_menu_load_calib(self):
        file = filedialog.askopenfile(
            title="Select Calibration file",
            filetypes=(("calibration files", "*.cal"), ("all files", "*.*")),
        )
        self.cal.calibrator.load(file.name)

    def on_menu_save_calib(self):
        default_name = self.cal.calibrator.selected_camera.name.replace(" ", "_")

        file = filedialog.asksaveasfile(
            title="Save Calibration file",
            filetypes=(("calibration files", "*.cal"), ("all files", "*.*")),
            defaultextension=".cal",
            initialfile=default_name,
        )
        self.cal.calibrator.save(file.name)
