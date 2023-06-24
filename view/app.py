import tkinter as tk
from tkinter import ttk
from model.scene import Scene
import numpy as np

from model.fanuc_crx10ial import FanucCRX10iAL
from model.camera.rs_d415 import RealsenseD415


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
        cam = RealsenseD415()
        cam.attach(crx, np.eye(4))  # TODO load from disk?
        self.scene.add_robot(crx)
        self.scene.add_camera(cam)

    def run(self):
        print("Running...")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        tabs = ttk.Notebook(self.root)
        tabs.grid(padx=10, pady=10, sticky=tk.NSEW)

        ov = Overview(tabs, self.scene)
        tabs.add(ov, text="Overview")

        cal = ViewCalibration(tabs, self.scene)
        tabs.add(cal, text="1. Camera Calibration")

        reg = ViewPoseRegistration(tabs, self.scene)
        tabs.add(reg, text="2. Pose Registration")

        acq = ViewAcquisition(tabs, self.scene)
        tabs.add(acq, text="3. Acquisition")

        tk.mainloop()
