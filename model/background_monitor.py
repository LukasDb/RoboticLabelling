import numpy as np
import tkinter as tk
from tkinter import ttk
from .entity import Entity
from PIL import Image, ImageTk

import cv2
from cv2 import aruco


class BackgroundMonitor(Entity):
    """Background Monitor, the pose refers to the center of the monitor
    with coordinate system as if it was a cv2 image
    """

    def __init__(self):
        # get secondary monitor info
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
        self.window.attributes("-fullscreen", True)
        self.window.update()
        self.width = self.window.winfo_width()
        self.height = self.window.winfo_height()

        # get screen name
        self.screen_name = self.window.winfo_screen()

        # get screen size
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        # get screen size in mm
        self.screen_width_mm = self.window.winfo_screenmmwidth()
        self.screen_height_mm = self.window.winfo_screenmmheight()

        # get screen position
        self.screen_x = self.window.winfo_x()
        self.screen_y = self.window.winfo_y()

        # check for successfull fullscreen:
        if self.width == self.screen_width and self.height == self.screen_height:
            print("Fullscreen successful")
        else:
            print("Fullscreen failed")
            return

        pixel_w = self.screen_width_mm / self.screen_width / 1000.0
        pixel_h = self.screen_height_mm / self.screen_height / 1000.0
        print(f"Pixel dimensions: {pixel_w} x {pixel_h} m")

        chessboardSize = 0.05 // pixel_w * pixel_w  # in m
        # pixel_size * 126  # old value: 0.023 # [m]
        markerSize = 0.04 // pixel_w * pixel_w
        n_markers = (7, 5)  # x,y

        # create an appropriately sized image
        charuco_img_width = n_markers[0] * chessboardSize  # in m
        charuco_img_width = (
            charuco_img_width / self.screen_width_mm * self.screen_width * 1000
        )  # in pixel

        charuco_img_height = n_markers[1] * chessboardSize  # in m
        charuco_img_height = (
            charuco_img_height / self.screen_height_mm * self.screen_height * 1000
        )

        charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        charuco_board = aruco.CharucoBoard.create(
            n_markers[0], n_markers[1], chessboardSize, markerSize, charuco_dict
        )
        print(
            "Creating Charuco image with size: ", charuco_img_width, charuco_img_height
        )
        charuco_img = charuco_board.draw(
            (round(charuco_img_width), round(charuco_img_height)))

        hor_pad = round((self.screen_width - charuco_img_width) / 2)
        vert_pad = round((self.screen_height - charuco_img_height) / 2)
        charuco_img = cv2.copyMakeBorder(
            charuco_img,
            vert_pad,
            vert_pad,
            hor_pad,
            hor_pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        self.set_image(charuco_img)

        print(
            f"Confirm the dimensions of the chessboard in the image: {chessboardSize}"
        )
        print(f"Confirm the dimensions of the markers in the image: {markerSize}")

    def set_image(self, image: np.ndarray):
        """Set the image of the background monitor"""
        self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.itemconfig(self.image_container, image=self.image_tk)
