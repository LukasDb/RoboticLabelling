import asyncio
import tkinter as tk
from tkinter import ttk
import logging
from PIL import ImageTk, Image
import numpy as np
import cv2
import time

from robolabel.scene import Scene
from robolabel.operators import CameraCalibrator
from ..lib.resizable_image import ResizableImage


class ViewCalibration(ttk.Frame):
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480

    def __init__(self, master, scene: Scene, calibrator: CameraCalibrator) -> None:
        ttk.Frame.__init__(self, master)
        self.scene = scene
        self.calibrator = calibrator

        self.title = ttk.Label(self, text="1. Camera Calibration")
        self.title.grid(row=0, column=0, columnspan=2)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0, minsize=200)
        self.rowconfigure(1, weight=1)

        self.control_frame = self.setup_controls(self)
        self.preview_frame = self.setup_previews()

        self.control_frame.grid(row=1, column=1, sticky=tk.NSEW)
        self.preview_frame.grid(row=1, column=0, sticky=tk.NSEW)

    def setup_controls(self, master):
        control_frame = ttk.Frame(master)

        self.btn_setup = ttk.Button(
            control_frame,
            text="Setup",
            command=self.calibrator.setup,
        )

        self.capture_button = ttk.Button(
            control_frame, text="Capture Image", command=self.on_capture
        )

        self.clear_button = ttk.Button(
            control_frame,
            text="Delete Captured Images",
            command=self.on_delete,
        )

        self.image_selection = ttk.Combobox(control_frame)
        self.image_selection.bind(
            sequence="<<ComboboxSelected>>", func=lambda _: self.update_selected_image_preview()
        )

        self.calibrate_button = ttk.Button(
            control_frame,
            text="Calibrate Intrinsics & Hand-Eye",
            command=self.on_calibrate,
        )

        def sb_pos() -> ttk.Spinbox:
            spinbox = ttk.Spinbox(
                control_frame,
                from_=-1.0,
                to=1.0,
                increment=0.01,
                command=lambda: self._change_initial_guess(),
            )
            # bind self._on_change to spinbox
            spinbox.bind("<Return>", lambda _: self._change_initial_guess())
            return spinbox

        self.guess_x = sb_pos()
        self.guess_y = sb_pos()
        self.guess_z = sb_pos()

        self.guess_x.set(0.367)
        self.guess_y.set(-0.529)
        self.guess_z.set(-0.16)

        self._change_initial_guess()

        pady = 5
        self.btn_setup.grid(pady=pady)
        self.capture_button.grid(pady=pady)
        self.clear_button.grid(pady=pady)
        self.image_selection.grid(pady=pady)
        self.guess_x.grid(pady=pady)
        self.guess_y.grid(pady=pady)
        self.guess_z.grid(pady=pady)
        self.calibrate_button.grid(pady=pady)

        return control_frame

    def setup_previews(self):
        preview_frame = ttk.Frame(self)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.selected_image_canvas = ResizableImage(preview_frame, bg="#000000")
        self.selected_image_canvas.grid(sticky=tk.NSEW)
        return preview_frame

    def update_selected_image_preview(self):
        selected = self.image_selection.get()
        try:
            selected_index = int(selected.split(" ")[-1])
        except ValueError:
            return

        img = self.calibrator.get_selected_img(selected_index)
        if img is not None:
            self.selected_image_canvas.set_image(img)

    def on_capture(self):
        def update_previews():
            if len(self.calibrator.captured_images) == 0:
                return
            self.image_selection["values"] = [
                f"Image {i:2}" for i in range(len(self.calibrator.captured_images))
            ]
            self.image_selection.set(self.image_selection["values"][-1])

            self.update_selected_image_preview()

        asyncio.get_event_loop().create_task(self.calibrator.capture_images(update_previews))

    def on_calibrate(self):
        self.calibrator.calibrate()
        self.update_selected_image_preview()

    def on_delete(self):
        self.calibrator.reset()
        self.image_selection["values"] = []
        self.image_selection.set("")
        self.selected_image_canvas.clear_image()

    def _change_initial_guess(self):
        try:
            x = float(self.guess_x.get())  # user inputs in spinbox
            y = float(self.guess_y.get())
            z = float(self.guess_z.get())
        except ValueError:
            logging.error("Must enter float values!")
            return

        self.calibrator.set_initial_guess(x, y, z)
