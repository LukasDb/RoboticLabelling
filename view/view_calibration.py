import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from model.scene import Scene
from control.camera_calibration import CameraCalibrator
import numpy as np


class ViewCalibration(tk.Frame):
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480

    def __init__(self, parent, scene: Scene) -> None:
        tk.Frame.__init__(self, parent)
        self.scene = scene

        self.title = ttk.Label(self, text="1. Camera Calibration")
        self.title.grid(
            row=0,
            column=0,
        )

        self.calibrator = calibrator = CameraCalibrator(self.scene)

        self.camera_selection = ttk.Combobox(
            self, values=[c.name for c in self.scene.cameras]
        )
        self.camera_selection.grid(row=1, column=1)

        self.capture_button = ttk.Button(
            self, text="Capture Image", command=self.on_capture
        )
        self.capture_button.grid(row=2, column=1)

        self.clear_button = ttk.Button(
            self,
            text="Delete Captured Images",
            command=calibrator.captured_images.clear,
        )
        self.clear_button.grid(row=3, column=1)

        self.image_selection = ttk.Combobox(self)
        self.image_selection.grid(row=3, column=1)
        self.image_selection.bind(
            "<<ComboboxSelected>>", lambda _: self.update_selected_image_preview()
        )

        self.calibrate_button = ttk.Button(
            self, text="Calibrate Intrinsics & Hand-Eye", command=calibrator.calibrate
        )
        self.calibrate_button.grid(row=4, column=1)


        self.live_canvas = tk.Canvas(self, width=self.PREVIEW_WIDTH, height=self.PREVIEW_HEIGHT)
        self.live_img_tk = ImageTk.PhotoImage(image=Image.fromarray(calibrator.get_live_img()))
        self.live_canvas.create_image(0,0, anchor=tk.NW, image=self.live_img_tk)
        self.live_canvas.grid(row=1, column=0)


        self.selected_image_canvas = tk.Canvas(self, width=self.PREVIEW_WIDTH, height=self.PREVIEW_HEIGHT)
        self.selected_image_container = self.selected_image_canvas.create_image(
            0, 0, anchor=tk.NW
        )
        self.selected_image_canvas.grid(row=2, column=0)

    def update_selected_image_preview(self):
        selected = self.image_selection.get()
        selected_index = int(selected.split(" ")[-1])

        img = self.calibrator.get_selected_img(selected_index)

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
        