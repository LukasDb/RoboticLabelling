import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from model.scene import Scene
from control.camera_calibration import CameraCalibrator
import numpy as np
import cv2


class ViewCalibration(ttk.Frame):
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480

    def __init__(self, parent, scene: Scene) -> None:
        ttk.Frame.__init__(self, parent)
        self.scene = scene

        self.title = ttk.Label(self, text="1. Camera Calibration")
        self.control_frame = ttk.Frame(self)
        self.preview_frame = ttk.Frame(self)

        self.title.grid(row=0, column=0, columnspan=2)
        self.control_frame.grid(row=1, column=1, sticky=tk.N)
        self.preview_frame.grid(row=1, column=0)

        self.calibrator = CameraCalibrator(self.scene, None)

        self.setup_controls()
        self.setup_previews()

    def setup_controls(self):
        self.btn_setup = ttk.Button(
            self.control_frame,
            text="Setup",
            command=self.calibrator.setup,
        )

        self.camera_selection = ttk.Combobox(
            self.control_frame, values=[c.name for c in self.scene.cameras]
        )
        self.camera_selection.bind(
            "<<ComboboxSelected>>", self.calibrator.select_camera
        )

        self.capture_button = ttk.Button(
            self.control_frame, text="Capture Image", command=self.on_capture
        )

        self.clear_button = ttk.Button(
            self.control_frame,
            text="Delete Captured Images",
            command=self.calibrator.captured_images.clear,
        )

        self.image_selection = ttk.Combobox(self.control_frame)
        self.image_selection.bind(
            "<<ComboboxSelected>>", lambda _: self.update_selected_image_preview()
        )

        self.calibrate_button = ttk.Button(
            self.control_frame,
            text="Calibrate Intrinsics & Hand-Eye",
            command=self.calibrator.calibrate,
        )

        self.btn_setup.grid(row=0)
        self.camera_selection.grid(row=1)
        self.capture_button.grid(row=2)
        self.clear_button.grid(row=3)
        self.image_selection.grid(row=4)
        self.calibrate_button.grid(row=5)

    def setup_previews(self):
        self.live_canvas = tk.Canvas(
            self.preview_frame, width=self.PREVIEW_WIDTH, height=self.PREVIEW_HEIGHT
        )
        self.selected_image_canvas = tk.Canvas(
            self.preview_frame, width=self.PREVIEW_WIDTH, height=self.PREVIEW_HEIGHT
        )

        self.live_img_tk = ImageTk.PhotoImage(
            image=Image.fromarray(self.calibrator.get_live_img())
        )
        self.live_canvas.create_image(0, 0, anchor=tk.NW, image=self.live_img_tk)
        self.selected_image_container = self.selected_image_canvas.create_image(
            0, 0, anchor=tk.NW
        )

        self.live_canvas.grid(row=0)
        self.selected_image_canvas.grid(row=1)

    def update_selected_image_preview(self):
        selected = self.image_selection.get()
        selected_index = int(selected.split(" ")[-1])

        img = self.calibrator.get_selected_img(selected_index)
        img = cv2.resize(img, (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT))

        if img is not None:
            self.sel_img_tk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.selected_image_canvas.itemconfig(
                self.selected_image_container, image=self.sel_img_tk
            )

    def on_capture(self):
        self.calibrator.capture_image()
        self.image_selection["values"] = [
            f"Image {i:2}" for i in range(len(self.calibrator.captured_images))
        ]
        self.image_selection.set(len(self.calibrator.captured_images) - 1)

        self.update_selected_image_preview()
