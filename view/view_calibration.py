import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
import cv2
import time

from model.scene import Scene
from model.observer import Observer, Event
from control.camera_calibration import CameraCalibrator
from view.resizable_image import ResizableImage



class ViewCalibration(Observer, ttk.Frame):
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480

    def __init__(self, parent, scene: Scene, calibrator: CameraCalibrator) -> None:
        ttk.Frame.__init__(self, parent)
        self.scene = scene
        self.listen_to(self.scene)
        self.calibrator = calibrator

        self.title = ttk.Label(self, text="1. Camera Calibration")
        self.title.grid(row=0, column=0, columnspan=2)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0, minsize=200)
        self.rowconfigure(1, weight=1)

        self.setup_controls()
        self.setup_previews()


    def update_observer(self, subject, event: Event, *args, **kwargs):
        if event == Event.CAMERA_ADDED:
            self.setup_controls()

    def setup_controls(self):
        self.control_frame = ttk.Frame(self)
        self.control_frame.grid(row=1, column=1, sticky=tk.NE)

        self.btn_setup = ttk.Button(
            self.control_frame,
            text="Setup",
            command=self.calibrator.setup,
        )

        self.capture_button = ttk.Button(
            self.control_frame, text="Capture Image", command=self.on_capture
        )

        self.clear_button = ttk.Button(
            self.control_frame,
            text="Delete Captured Images",
            command=self.on_delete,
        )

        self.image_selection = ttk.Combobox(self.control_frame)
        self.image_selection.bind(
            "<<ComboboxSelected>>", lambda _: self.update_selected_image_preview()
        )

        self.calibrate_button = ttk.Button(
            self.control_frame,
            text="Calibrate Intrinsics & Hand-Eye",
            command=self.on_calibrate,
        )

        pady = 5
        self.btn_setup.grid(pady=pady)
        self.capture_button.grid(pady=pady)
        self.clear_button.grid(pady=pady)
        self.image_selection.grid(pady=pady)
        self.calibrate_button.grid(pady=pady)

    def setup_previews(self):
        self.preview_frame = ttk.Frame(self)
        self.preview_frame.grid(row=1, column=0, sticky=tk.NSEW)
        self.preview_frame.rowconfigure(0, weight=1)
        self.preview_frame.rowconfigure(1, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)

        self.selected_image_canvas = ResizableImage(self.preview_frame, bg="#000000")

        self.selected_image_canvas.grid(row=1, sticky=tk.NSEW)


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
        self.calibrator.capture_image()
        self.image_selection["values"] = [
            f"Image {i:2}" for i in range(len(self.calibrator.captured_images))
        ]
        self.image_selection.set(self.image_selection["values"][-1])

        self.update_selected_image_preview()

    def on_calibrate(self):
        self.calibrator.calibrate()
        self.update_selected_image_preview()

    def on_delete(self):
        self.calibrator.captured_images.clear()
        self.image_selection["values"] = []
        self.image_selection.set("")
        self.selected_image_canvas.clear_image()

