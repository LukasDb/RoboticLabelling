from .camera import Camera, CamFrame
import numpy as np
import cv2
from pathlib import Path
from itertools import cycle
import time
import tkinter as tk
from tkinter import ttk


class DemoCam(Camera):
    FPS = 2

    def __init__(self, unique_id: str):
        super().__init__("MockCam")
        self._unique_id = unique_id
        # self.mock_cam = "realsense_121622061798"
        # self.mock_cam = "realsense_f1120593" #  old
        # self.mock_cam = "0_Intel RealSense D415"
        # self.mock_cam = "1_Intel RealSense D415"
        self.data_folder = None
        self.img_paths = None
        self.img_index = 0
        self.n_images = 0
        self.last_t = time.time()
        # self.img_paths = cycle(Path(f"demo_data/images/{self.mock_cam}").glob("*.png"))

        self.window = tk.Toplevel()
        self.window.title("Demo Camera Controls")
        self.window.attributes("-topmost", True)
        self.window.attributes("-type", "dialog")

        # selectbox
        self.selectbox = ttk.Combobox(self.window, values=list(Path("demo_data").iterdir()))
        self.selectbox.bind("<<ComboboxSelected>>", lambda _: self._on_data_selected())
        self.selectbox.grid(row=0, column=0, sticky=tk.NSEW)

        # checkbox to toggle play
        self.auto_play = False
        self.checkbox = ttk.Checkbutton(
            self.window, text="Auto Play", command=lambda: self._toggle_auto_play()
        )
        self.checkbox.grid(row=1, column=0, sticky=tk.NSEW)

        # select slider for image index
        self.slider = tk.Scale(
            self.window,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._on_slider_change,
            tickinterval=10,
            resolution=1,
        )
        self.slider.grid(row=2, column=0, sticky=tk.NSEW)

    def _on_slider_change(self, value):
        with self.lock:
            self.img_index = int(value)

    def _toggle_auto_play(self):
        with self.lock:
            self.auto_play = not self.auto_play

    def _on_data_selected(self):
        # updatet indices for slider
        with self.lock:
            self.data_folder = self.selectbox.get()
            img_path = Path(f"{self.data_folder}/images").glob("*.png")
            self.n_images = len(list(img_path))
            self.slider.configure(to=self.n_images - 1)
            self.slider.set(0)

    def get_frame(self) -> CamFrame:
        with self.lock:
            frame = self._get_frame()
        return frame

    def _get_frame(self) -> CamFrame:
        if self.data_folder is None:
            return CamFrame()

        if self.auto_play:
            if time.time() - self.last_t > 1 / self.FPS:
                self.img_index += 1
                self.img_index %= self.n_images
                self.slider.set(self.img_index)
                self.last_t = time.time()

        # img_path = next(self.img_paths)
        img_path = Path(f"{self.data_folder}/images/{self.img_index}.png")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = None

        try:
            depth = np.load(f"{self.data_folder}/depth/{self.img_index}.npz.npy")
        except Exception as e:
            pass

        pose_path = Path(f"{self.data_folder}/poses/{self.img_index}.txt")
        robot_pose = np.loadtxt(str(pose_path))

        if "0_Intel RealSense D415" in self.data_folder and self._link_matrix is not None:
            # accidentally saved camera pose instead of robot pose
            robot_pose = robot_pose @ np.linalg.inv(self._link_matrix)

        # update mock robot pose
        if self.parent is not None:
            self.parent.pose = robot_pose
        return CamFrame(rgb=img, depth=depth)

    @property
    def unique_id(self) -> str:
        return self._unique_id
