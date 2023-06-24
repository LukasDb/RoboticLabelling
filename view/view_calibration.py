import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from model.scene import Scene
from control.camera_calibration import CameraCalibrator
import numpy as np
import cv2
from view.resizable_image import ResizableImage
import threading


class ViewCalibration(ttk.Frame):
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480

    def __init__(self, parent, scene: Scene) -> None:
        ttk.Frame.__init__(self, parent)
        self.scene = scene

        self.title = ttk.Label(self, text="1. Camera Calibration")
        self.title.grid(row=0, column=0, columnspan=2)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0, minsize=200)
        self.rowconfigure(1, weight=1)

        self.calibrator = CameraCalibrator(self.scene)

        self.setup_controls()
        self.setup_previews()

        self.stop_event = threading.Event()
        self.live_thread = threading.Thread(target=self.live_thread_fn)
        self.live_thread.start()

    def destroy(self) -> None:
        self.stop_event.set()
        return super().destroy()

    def setup_controls(self):
        self.control_frame = ttk.Frame(self)
        self.control_frame.grid(row=1, column=1, sticky=tk.NE)

        self.btn_setup = ttk.Button(
            self.control_frame,
            text="Setup",
            command=self.calibrator.setup,
        )

        self.camera_selection = ttk.Combobox(
            self.control_frame, values=[c.name for c in self.scene.cameras.values()]
        )
        self.camera_selection.set(self.calibrator.selected_camera.name)
        self.camera_selection.bind(
            "<<ComboboxSelected>>", self.calibrator.select_camera
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

        self.btn_setup.grid(row=0)
        self.camera_selection.grid(row=1)
        self.capture_button.grid(row=2)
        self.clear_button.grid(row=3)
        self.image_selection.grid(row=4)
        self.calibrate_button.grid(row=5)

    def setup_previews(self):
        self.preview_frame = ttk.Frame(self)
        self.preview_frame.grid(row=1, column=0, sticky=tk.NSEW)
        self.preview_frame.rowconfigure(0, weight=1)
        self.preview_frame.rowconfigure(1, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)

        live_img = self.calibrator.get_live_img()
        self.live_canvas = ResizableImage(
            self.preview_frame, image=live_img, bg="#000000"
        )

        self.selected_image_canvas = ResizableImage(self.preview_frame, bg="#000000")

        self.live_canvas.grid(row=0, sticky=tk.NSEW)
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

    def live_thread_fn(self):
        while not self.stop_event.is_set():
            try:
                img = self.calibrator.get_live_img()
                self.live_canvas.set_image(img)
            except Exception:
                pass