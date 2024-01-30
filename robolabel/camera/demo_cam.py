from .camera import Camera, CamFrame, DepthQuality
import numpy as np
import asyncio
import logging
import cv2
from pathlib import Path
from itertools import cycle
import time
import tkinter as tk
from tkinter import ttk
from robolabel.lib.geometry import invert_homogeneous, distance_from_matrices


class DemoCam(Camera):
    FPS = 2

    def __init__(self, unique_id: str):
        super().__init__("MockCam")
        self._unique_id = unique_id
        self.data_folder = None
        self.img_paths = None
        self.img_index = 0
        self.n_images = 0
        self.last_t = time.time()

        self.window = tk.Toplevel()
        self.window.title("Demo Camera Controls")
        self.window.attributes("-topmost", True)
        self.window.attributes("-type", "dialog")

        # selectbox
        available_data = list(str(x) for x in Path("demo_data").iterdir())
        self.selectbox = ttk.Combobox(self.window, values=available_data)
        self.selectbox.bind("<<ComboboxSelected>>", lambda _: self._on_data_selected())
        self.selectbox.grid(row=0, column=0, sticky=tk.NSEW)
        self.selectbox.set(available_data[0])

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

        self._on_data_selected()

    def _on_slider_change(self, value):
        self.img_index = int(value)

    def _toggle_auto_play(self):
        self.auto_play = not self.auto_play

    def _on_data_selected(self):
        # update indices for slider
        self.selectbox.configure(values=list(str(x) for x in Path("demo_data").iterdir()))
        self.data_folder = self.selectbox.get()
        img_path = Path(f"{self.data_folder}/images").glob("*.png")
        self.n_images = len(list(img_path))
        self.slider.configure(to=self.n_images - 1)
        self.slider.set(0)

        self.robot_poses = {
            int(pose_path.stem): np.loadtxt(str(pose_path))
            for pose_path in Path(f"{self.data_folder}/poses/").glob("*.txt")
        }

        frame = self.get_frame(depth_quality=DepthQuality.INFERENCE)
        assert frame.rgb is not None
        self.width = frame.rgb.shape[1]
        self.height = frame.rgb.shape[0]

    def get_frame(self, depth_quality: DepthQuality) -> CamFrame:
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
        img = cv2.imread(str(img_path))  # type: ignore
        if img is None:
            return CamFrame()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore
        depth = None

        try:
            depth = np.load(f"{self.data_folder}/depth/{self.img_index}.npz.npy")
        except Exception as e:
            pass

        robot_pose = self.robot_poses[self.img_index]

        if "Intel RealSense D415_0" in self.data_folder and self._extrinsics is not None:
            # accidentally saved camera pose instead of robot pose
            robot_pose = robot_pose @ invert_homogeneous(self._extrinsics)

        # update mock robot pose
        if self.robot is not None:
            self.robot.pose = robot_pose

            # whenever we move the robot, choose a corresponding image
            async def move_to_with_cb(pose, timeout):
                dists = {distance_from_matrices(pose, p): i for i, p in self.robot_poses.items()}
                min_dist = np.min(list(dists.keys()))
                closest = dists[min_dist]
                self.img_index = closest
                self.slider.set(closest)
                await asyncio.sleep(0.5)
                logging.info(f"{self.name} would moved to pose: {pose[:3, 3]}")
                return True

            self.robot.move_to = move_to_with_cb
        return CamFrame(rgb=img, depth=depth)

    @property
    def unique_id(self) -> str:
        return self._unique_id
