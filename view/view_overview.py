import tkinter as tk
from tkinter import ttk
import threading
from typing import Dict
import numpy as np
import time

from model.scene import Scene
from model.camera.camera import Camera
from model.observer import Observer, Event
from view.resizable_image import ResizableImage
from control.camera_calibration import CameraCalibrator


class Overview(Observer, tk.Frame):
    def __init__(self, parent, scene: Scene, calibrator: CameraCalibrator) -> None:
        # TODO semantics editor: create an object, by selecting a mesh and a label
        # TODO show available cameras
        tk.Frame.__init__(self, parent)
        Observer.__init__(self)
        self._scene = scene
        self._calibrator = calibrator
        self.listen_to(self._scene)

        self._cam_rows: Dict[str, Dict] = {}
        self._object_rows: Dict[str, Dict] = {}

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
        if event == Event.CAMERA_CALIBRATED or event == Event.CAMERA_ATTACHED:
            # update calibrated column
            self._update_cam_row(subject.unique_id)

        elif event == Event.CAMERA_ADDED:
            self.listen_to(kwargs["camera"])
            self._add_camera_row(kwargs["camera"])
            # update camera selection values
            self.camera_selection.configure(
                values=[c.unique_id for c in self._scene.cameras.values()]
            )

        elif event == Event.OBJECT_ADDED:
            self._add_object_row(kwargs["object"])

        elif event == Event.ROBOT_ADDED:
            for cam in self._scene.cameras.values():
                self._update_cam_row(cam.unique_id)

    def setup_controls(self, parent):
        control_frame = tk.Frame(parent)
        control_frame.columnconfigure(0, weight=1)

        cam_select_frame = tk.Frame(control_frame)
        self.camera_selection = ttk.Combobox(
            cam_select_frame, values=[c.unique_id for c in self._scene.cameras.values()]
        )
        self.camera_selection.set("")
        self.camera_selection.bind(
            "<<ComboboxSelected>>", self._on_camera_selection_change
        )
        camera_selection_label = tk.Label(cam_select_frame, text="Selected Camera")
        camera_selection_label.grid(row=0, column=0, padx=5, sticky=tk.NW)
        self.camera_selection.grid(row=0, column=1, sticky=tk.NW)

        self.cam_overview = self.setup_camera_table(control_frame)
        self.object_overview = self.setup_object_table(control_frame)

        cam_select_frame.grid(sticky=tk.NSEW, pady=10)
        self.cam_overview.grid(sticky=tk.NSEW, pady=10)
        self.object_overview.grid(sticky=tk.NSEW, pady=10)
        return control_frame

    def setup_preview(self, parent):
        preview_frame = ttk.Frame(parent)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.live_canvas = ResizableImage(preview_frame, bg="#000000")
        self.live_canvas.grid(sticky=tk.NSEW)
        return preview_frame

    def setup_camera_table(self, parent):
        self.cam_overview = tk.Frame(parent)
        self.cam_overview.columnconfigure(0, weight=1)
        self.cam_overview.columnconfigure(1, weight=1)
        self.cam_overview.columnconfigure(2, weight=1)

        columns = ["Camera", "Calibrated?", "Attached"]
        for i, c in enumerate(columns):
            # headings.columnconfigure(i, weight=1)
            label = tk.Label(self.cam_overview, text=c)
            label.grid(row=0, column=i, sticky=tk.EW)

        # add separator
        sep = ttk.Separator(self.cam_overview, orient=tk.HORIZONTAL)
        sep.grid(column=0, sticky=tk.EW, columnspan=3)

        for c in self._scene.cameras.values():
            self._add_camera_row(c)

        return self.cam_overview

    def _add_camera_row(self, c):
        # add camera name
        row_num = len(self._cam_rows) + 2  # because of separator

        cam_name = tk.Label(self.cam_overview)
        cam_name.grid(row=row_num, column=0, sticky=tk.EW, pady=2)
        # add calibrated
        calibrated = tk.Label(self.cam_overview)
        calibrated.grid(row=row_num, column=1, sticky=tk.EW)
        # add parent
        # parent = tk.Label(row)
        parent = ttk.Combobox(
            self.cam_overview, values=["-"] + list(self._scene.robots.keys())
        )
        parent.set("-")
        parent.bind(
            "<<ComboboxSelected>>",
            lambda event, cam_unique_id=c.unique_id, parent=parent: self._on_attach_cam(
                cam_unique_id, parent
            ),
        )
        parent.grid(row=row_num, column=2, sticky=tk.EW)

        self._cam_rows[c.unique_id] = {
            "cam_name": cam_name,
            "calibrated": calibrated,
            "parent": parent,
        }

        # update values
        self._update_cam_row(c.unique_id)

    def _update_cam_row(self, cam_unique_id):
        cam = self._scene.cameras[cam_unique_id]
        row = self._cam_rows[cam_unique_id]
        row["cam_name"].configure(text=f"{cam.name} ({cam.unique_id})")
        is_calibrated = cam.intrinsic_matrix is not None
        row["calibrated"].configure(
            text="Yes" if is_calibrated else "No",
            fg="green" if is_calibrated else "red",
        )
        row["parent"].set("-" if cam.parent is None else cam.parent.name)
        row["parent"].configure(values=["-"] + list(self._scene.robots.keys()))

    def setup_object_table(self, parent):
        self.object_overview = tk.Frame(parent)
        self.object_overview.columnconfigure(0, weight=1)
        self.object_overview.columnconfigure(1, weight=1)

        columns = ["Object", "Plyfile"]
        for i, c in enumerate(columns):
            label = tk.Label(self.object_overview, text=c)
            label.grid(row=0, column=i, sticky=tk.EW)
        # add separator
        sep = ttk.Separator(self.object_overview, orient=tk.HORIZONTAL)
        sep.grid(row=1, column=0, sticky=tk.EW, columnspan=2)

        for c in self._scene.objects.values():
            self._add_object_row(c)

        return self.object_overview

    def _add_object_row(self, c):
        # add object name
        row_num = len(self._object_rows) + 2  # because of separator
        obj_name = tk.Label(self.object_overview)
        obj_name.grid(row=row_num, column=0, sticky=tk.EW, pady=2)
        # add pose
        plyfile = tk.Label(self.object_overview)
        plyfile.grid(row=row_num, column=1, sticky=tk.EW)

        self._object_rows[c.name] = {
            "obj_name": obj_name,
            "plyfile": plyfile,
        }
        # update values
        self._update_object_row(c.name)

    def _update_object_row(self, obj_name):
        obj = self._scene.objects[obj_name]
        row = self._object_rows[obj_name]
        row["obj_name"].configure(text=f"{obj.name}")
        row["plyfile"].configure(text=f"{str(obj.mesh_path)}")

    def _on_attach_cam(self, cam_unique_id, parent):
        if parent.get() == "-":
            self._scene.cameras[cam_unique_id].detach()
        else:
            robot = self._scene.robots[parent.get()]
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
                continue

            frame = selected_cam.get_frame()
            img = frame.rgb

            img = self._calibrator.draw_calibration(img)

            if img is not None:
                self.live_canvas.set_image(img)
            else:
                self.live_canvas.clear_image()
