import numpy as np
import tkinter as tk
from tkinter import ttk
from .entity import Entity
from PIL import Image, ImageTk


class BackgroundMonitor(Entity):
    """Background Monitor, the pose refers to the center of the monitor
    with coordinate system as if it was a cv2 image
    """

    def __init__(self):
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
        #self.window.attributes("-fullscreen", True)
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
            self.is_setup = True
        else:
            print("Fullscreen failed")

    def set_image(self, image: np.ndarray):
        """Set the image of the background monitor"""
        self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.itemconfig(self.image_container, image=self.image_tk)
