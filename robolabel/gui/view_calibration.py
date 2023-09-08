import asyncio
import tkinter as tk
from tkinter import ttk
import logging
from PIL import ImageTk, Image
import numpy as np
import cv2
import time

import robolabel as rl


class ViewCalibration(ttk.Frame):
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 480

    def __init__(self, master, scene: rl.Scene, calibrator: rl.operators.CameraCalibrator) -> None:
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

        self.capture_button = ttk.Button(control_frame, text="Capture", command=self.capture_image)

        self.automatic_button = ttk.Button(
            control_frame,
            text="Auto Capture",
            command=self.on_automatic_acquisition,
        )
        self.cancel_button = ttk.Button(
            control_frame,
            text="Cancel",
            command=lambda: self.calibrator.acquisition.cancel(),
        )

        self.clear_button = ttk.Button(
            control_frame,
            text="Reset",
            command=self.on_delete,
        )

        self.image_selection = ttk.Combobox(control_frame)
        self.image_selection.bind(
            sequence="<<ComboboxSelected>>", func=lambda _: self._update_gui()
        )

        self.self_calibrate_button = ttk.Button(
            control_frame,
            text="Run self-calibration",
            command=self._on_self_calibrate,
        )
        self.calibrate_button = ttk.Button(
            control_frame,
            text="Calibrate Intrinsics & Hand-Eye",
            command=self.calibrator.calibrate,
        )

        guess_frame = ttk.Frame(control_frame)

        def sb_pos() -> ttk.Spinbox:
            spinbox = ttk.Spinbox(
                guess_frame,
                from_=-1.0,
                to=1.0,
                increment=0.01,
                width=5,
                command=lambda: self._change_initial_guess(),
            )
            # bind self._on_change to spinbox
            spinbox.bind("<Return>", lambda _: self._change_initial_guess())
            return spinbox

        self.guess_x = sb_pos()
        self.guess_y = sb_pos()
        self.guess_z = sb_pos()

        pady = 5
        self.guess_x.grid(pady=pady, row=0, column=0)
        self.guess_y.grid(pady=pady, row=0, column=1)
        self.guess_z.grid(pady=pady, row=0, column=2)

        self.guess_x.set(0.367)
        self.guess_y.set(-0.529)
        self.guess_z.set(-0.16)

        self._change_initial_guess()

        self.btn_setup.grid(pady=pady, row=0, column=0)
        self.capture_button.grid(pady=pady, row=0, column=1)

        self.automatic_button.grid(pady=pady, row=1, column=0)
        self.cancel_button.grid(pady=pady, row=1, column=1)

        self.image_selection.grid(pady=pady, row=2, column=0)
        self.clear_button.grid(pady=pady, row=2, column=1)

        guess_frame.grid(pady=pady, row=3, column=0, columnspan=3)

        self.calibrate_button.grid(pady=pady, row=4, column=0, columnspan=3)
        self.self_calibrate_button.grid(pady=pady, row=5, column=0, columnspan=3)

        return control_frame

    def setup_previews(self):
        preview_frame = ttk.Frame(self)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.selected_image_canvas = rl.ResizableImage(preview_frame, bg="#000000")
        self.selected_image_canvas.grid(sticky=tk.NSEW)
        return preview_frame

    @rl.as_async_task
    async def _update_gui(self, set_to_last_image=False):
        if len(self.calibrator.calibration_datapoints) == 0:
            return
        self.image_selection["values"] = [
            f"Image {i:2}" for i in range(len(self.calibrator.calibration_datapoints))
        ]
        if set_to_last_image:
            self.image_selection.set(self.image_selection["values"][-1])

        selected = self.image_selection.get()
        try:
            selected_index = int(selected.split(" ")[-1])
        except ValueError:
            return

        img = await self.calibrator.get_from_image_cache(selected_index)
        if img is not None:
            self.selected_image_canvas.set_image(img)

    @rl.as_async_task
    async def capture_image(self):
        await self.calibrator.capture()
        self._update_gui(set_to_last_image=True)

    def on_automatic_acquisition(self):
        # check if asyncio task is already running
        if "auto_calibration" in [t.get_name() for t in asyncio.all_tasks()]:
            logging.warn("Auto calibration already running!")
            return

        asyncio.get_event_loop().create_task(
            self.calibrator.capture_images(lambda: self._update_gui(set_to_last_image=True)),
            name="auto_calibration",
        )

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

    @rl.as_async_task
    async def _on_self_calibrate(self):
        await self.calibrator.run_self_calibration()
        self._update_gui(set_to_last_image=True)
