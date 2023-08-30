import tkinter as tk
from tkinter import ttk
from tkinter.colorchooser import askcolor
import threading
from pathlib import Path
from typing import Dict
import logging
import numpy as np
import time
import cv2

from robolabel.scene import Scene
from robolabel.labelled_object import LabelledObject
from robolabel.observer import Observer, Event
from .resizable_image import ResizableImage
from .widget_list import WidgetList
from robolabel.operators import CameraCalibrator, PoseRegistrator


class Overview(Observer, tk.Frame):
    def __init__(
        self,
        parent,
        scene: Scene,
        calibrator: CameraCalibrator,
        registrator: PoseRegistrator,
    ) -> None:
        tk.Frame.__init__(self, parent)
        Observer.__init__(self)
        self._scene = scene
        self._calibrator = calibrator
        self._registrator = registrator
        self.listen_to(self._scene)

        self.cam_list: None | WidgetList = None
        self.object_list: None | WidgetList = None

        self.title = ttk.Label(self, text="Overview")
        self.controls = self.setup_controls(self)
        self.preview = self.setup_preview(self)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=0, minsize=30)
        self.rowconfigure(1, weight=1)

        self.title.grid(columnspan=2)
        self.preview.grid(column=0, row=1, sticky=tk.NSEW)
        self.controls.grid(column=1, row=1, sticky=tk.NSEW)

        self.stop_event = threading.Event()
        self.live_thread = threading.Thread(target=self.live_thread_fn, daemon=True)
        self.live_thread.start()

    def destroy(self) -> None:
        self.stop_event.set()
        return super().destroy()

    def update_observer(self, subject, event: Event, *args, **kwargs):
        if (
            event == Event.CAMERA_CALIBRATED
            or event == Event.CAMERA_ATTACHED
            or event == Event.CAMERA_ADDED
            or event == Event.ROBOT_ADDED
        ):
            # update calibrated column
            self._update_cam_table()
            for cam in self._scene.cameras.values():
                self.listen_to(cam)
            self.camera_selection.configure(
                values=[c.unique_id for c in self._scene.cameras.values()]
            )

        elif (
            event == Event.OBJECT_REMOVED
            or event == Event.OBJECT_REGISTERED
            or event == Event.OBJECT_ADDED
        ):
            self._update_object_table()
            for obj in self._scene.objects.values():
                self.listen_to(obj)

    def setup_controls(self, parent):
        control_frame = tk.Frame(parent)
        control_frame.columnconfigure(0, weight=1)

        cam_select_frame = tk.Frame(control_frame)
        self.camera_selection = ttk.Combobox(
            cam_select_frame, values=[c.unique_id for c in self._scene.cameras.values()]
        )
        self.camera_selection.set("")
        self.camera_selection.bind("<<ComboboxSelected>>", self._on_camera_selection_change)
        camera_selection_label = tk.Label(cam_select_frame, text="Selected Camera")
        camera_selection_label.grid(row=0, column=0, padx=5, sticky=tk.NW)
        self.camera_selection.grid(row=0, column=1, sticky=tk.NW)

        capture_frame = ttk.Frame(control_frame)
        capture_button = ttk.Button(capture_frame, text="Capture Image", command=self._on_capture)
        self.capture_name = ttk.Entry(capture_frame)
        self.capture_name.grid(row=0, column=0)
        capture_button.grid(row=0, column=1)

        self.cam_list = WidgetList(
            control_frame,
            column_names=["Camera", "Calibrated?", "Attached"],
            columns=[tk.Label, tk.Label, ttk.Combobox],
        )
        self.object_list = WidgetList(
            control_frame,
            column_names=["Object", "Registered?", "Color", "Remove"],
            columns=[tk.Label, tk.Label, tk.Label, ttk.Button],
        )

        cam_select_frame.grid(sticky=tk.NSEW, pady=10)
        capture_frame.grid(sticky=tk.NSEW, pady=10)
        self.cam_list.grid(sticky=tk.NSEW, pady=10)
        self.object_list.grid(sticky=tk.NSEW, pady=10)
        return control_frame

    def setup_preview(self, parent):
        preview_frame = ttk.Frame(parent)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.live_canvas = ResizableImage(preview_frame, bg="#000000")
        self.live_canvas.grid(sticky=tk.NSEW)
        self.live_canvas2 = ResizableImage(preview_frame, bg="#000000")
        self.live_canvas2.grid(sticky=tk.NSEW)
        return preview_frame

    def _on_capture(self):
        # save image for use in demo cam + robot pose
        data_name = self.capture_name.get()
        if data_name == "":
            logging.warn("Please enter a name for the data")
            return
        data_folder = Path(f"demo_data/{data_name}")

        if not data_folder.exists():
            data_folder.mkdir(parents=True, exist_ok=True)
            (data_folder / "images").mkdir(parents=True, exist_ok=True)
            (data_folder / "depth").mkdir(parents=True, exist_ok=True)
            (data_folder / "poses").mkdir(parents=True, exist_ok=True)

        cam = self._scene.selected_camera
        idx = len(list(data_folder.glob("images/*.png")))

        frame = cam.get_frame()
        if cam.parent is not None:
            robot_pose = cam.parent.pose
            np.savetxt(str(data_folder / f"poses/{idx}.txt"), robot_pose)
        if frame.rgb is not None:
            cv2.imwrite(
                str(data_folder / f"images/{idx}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)
            )
        if frame.depth is not None:
            np.save(str(data_folder / f"depth/{idx}.npy"), frame.depth)

    def _update_cam_table(self) -> None:
        self.cam_list.clear()
        for cam in self._scene.cameras.values():
            is_calibrated = cam.intrinsic_matrix is not None
            kwargs_list = [
                {"text": f"{cam.name} ({cam.unique_id})"},
                {
                    "text": "Yes" if is_calibrated else "No",
                    "fg": "green" if is_calibrated else "red",
                },
                {"values": ["-"] + list(self._scene.robots.keys())},
            ]

            _, _, w_parent = self.cam_list.add_new_row(kwargs_list)
            w_parent.bind(
                "<<ComboboxSelected>>",
                lambda event, cam_unique_id=cam.unique_id, parent=w_parent: self._on_attach_cam(
                    cam_unique_id, parent
                ),
            )

            w_parent.set("-" if cam.parent is None else cam.parent.name)

    def _update_object_table(self) -> None:
        self.object_list.clear()

        for obj in self._scene.objects.values():
            kwargs_list = [
                {"text": obj.name},  # obj_name
                {
                    "text": "Yes" if obj.registered else "No",
                    "fg": "green" if obj.registered else "red",
                },  # registered
                {"bg": "#%02x%02x%02x" % tuple(obj.semantic_color)},  # w_color
                {"text": "x", "command": lambda obj=obj: self._scene.remove_object(obj)},
            ]
            _, _, w_color, _ = self.object_list.add_new_row(kwargs_list)
            w_color.bind(
                "<Button-1>",
                lambda _: self._on_object_color_click(obj),
            )

    def _on_attach_cam(self, cam_unique_id, w_parent):
        if w_parent.get() == "-":
            self._scene.cameras[cam_unique_id].detach()
        else:
            robot = self._scene.robots[w_parent.get()]
            link_mat = self._scene.cameras[cam_unique_id].extrinsic_matrix
            link_mat = link_mat if link_mat is not None else np.eye(4)
            self._scene.cameras[cam_unique_id].attach(robot, link_mat)

    def _on_camera_selection_change(self, _):
        selected = self.camera_selection.get()
        self._scene.select_camera_by_id(selected)

    def live_thread_fn(self):
        t_previous = time.perf_counter()
        FPS = 20
        while not self.stop_event.is_set():
            t = time.perf_counter()
            if (t - t_previous) < 1 / FPS:  # 30 FPS
                time.sleep(1 / FPS - (t - t_previous))
            t_previous = time.perf_counter()

            selected_cam = self._scene.selected_camera

            if selected_cam is None:
                self.live_canvas.clear_image()
                self.live_canvas2.clear_image()
                continue

            frame = selected_cam.get_frame()
            img = frame.rgb

            if img is not None:
                img = self._calibrator.draw_calibration(img)
                img = self._registrator.draw_registered_objects(
                    img,
                    selected_cam.pose,
                    selected_cam.intrinsic_matrix,
                    selected_cam.dist_coeffs,
                )

                self.live_canvas.set_image(img)
            else:
                self.live_canvas.clear_image()

            if frame.depth is not None:
                colored_depth = cv2.applyColorMap(
                    cv2.convertScaleAbs(frame.depth, alpha=255 / 2.0), cv2.COLORMAP_JET
                )
                self.live_canvas2.set_image(colored_depth)
            else:
                self.live_canvas2.clear_image()

            if frame.rgb_R is not None:
                self.live_canvas2.set_image(frame.rgb_R)

    def _on_object_color_click(self, obj: LabelledObject):
        colors = askcolor(title="Object Semantic Color")
        obj.semantic_color = colors[0]
        self._update_object_row(obj.name)
