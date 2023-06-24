import numpy as np
import tkinter as tk
from tkinter import ttk
from .entity import Entity
from PIL import Image, ImageTk
import cv2
from lib.geometry import *


class BackgroundMonitor(Entity):
    """Background Monitor, the pose refers to the center of the monitor
    with coordinate system as if it was a cv2 image
    """

    def __init__(self):
        super().__init__("Background Monitor")
        # get secondary monitor info
        self.is_setup = False

        self.window = tk.Toplevel()
        self.window.title("Background Monitor -> MOVE THIS WINDOW TO SECONDARY MONITOR")

        # create fullscreen canvas
        self.canvas = tk.Canvas(self.window)
        # add image container
        self.image_container = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # move window to second screen
        self.window.geometry("640x480")
        self.window.geometry("+0+0")

    def setup_window(self):
        # get window size
        self.window.geometry("+0+0")
        # self.window.attributes("-fullscreen", True)
        self.window.update()
        self.width = self.window.winfo_width()
        self.height = self.window.winfo_height()

        # get screen name
        self.screen_name = self.window.winfo_screen()

        # get screen size
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        # get screen size in mm
        self.screen_width_m = self.window.winfo_screenmmwidth() / 1000.0
        self.screen_height_m = self.window.winfo_screenmmheight() / 1000.0

        # get screen position
        self.screen_x = self.window.winfo_x()
        self.screen_y = self.window.winfo_y()

        # MOCK FOR NOW
        if True:
            # overwrite to monitor from lab
            print("\nUSING MOCK MONITOR DIMENSIONS\n")
            self.screen_height = 2160
            self.screen_width = 3840
            diagonal_16_by_9 = np.linalg.norm((16, 9))
            self.screen_width_m = 16 / diagonal_16_by_9 * 0.800
            self.screen_height_m = 9 / diagonal_16_by_9 * 0.800

            print("screen_width_m", self.screen_width_m)
            print("screen_height_m", self.screen_height_m)

        # check for successfull fullscreen:
        if self.width == self.screen_width and self.height == self.screen_height:
            print("Fullscreen successful")
            self.is_setup = True
        else:
            print("Fullscreen failed")

    def set_image(self, image: np.ndarray):
        """Set the image of the background monitor"""
        self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.itemconfig(self.image_container, image=self.image_tk)

    def draw_on_rgb(self, rgb, intrinsic, dist_coeffs, cam2monitor, color = (255, 0, 0)):
        """Draw the background monitor on the rgb image"""
        half_w = self.screen_width_m / 2.0
        half_h = self.screen_height_m / 2.0

        screen_corners = np.array(
            [
                [-half_w, -half_h, 0, 1],
                [half_w, -half_h, 0, 1],
                [half_w, half_h, 0, 1],
                [-half_w, half_h, 0, 1],
            ]
        )
        screen_corners_cam = cam2monitor @ screen_corners.T
        # screen_corners_world = self.pose @ screen_corners.T
        rvec, tvec = get_rvec_tvec_from_affine_matrix(cam2monitor)
        img_points, _ = cv2.projectPoints(
            screen_corners_cam.T[:, :3],
            np.zeros(
                3,
            ),
            np.zeros(
                3,
            ),
            intrinsic,
            dist_coeffs,
        )

        cv2.drawContours(
            rgb,
            [img_points.astype(np.int32)],
            0,
            color,
            2,
        )

        # extrinsic==world2cam
        # cv2.drawFrameAxes(
        #     rgb,
        #     intrinsic,
        #     dist_coeffs,
        #     rvec,
        #     tvec,
        #     0.2,
        # )