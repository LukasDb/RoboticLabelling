import tkinter as tk
import asyncio
from tkinter import ttk
from tkinter.colorchooser import askcolor
from pathlib import Path
from typing import Dict
import logging
import numpy as np
import time
import cv2
import dataclasses

import robolabel as rl
from robolabel import Event


class Overview(rl.Observer, tk.Frame):
    def __init__(
        self,
        master,
        scene: rl.Scene,
        calibrator: rl.operators.CameraCalibrator,
        registration: rl.operators.PoseRegistration,
    ) -> None:
        tk.Frame.__init__(self, master)
        rl.Observer.__init__(self)
        self._scene = scene
        self._calibrator = calibrator
        self._registration = registration
        self.listen_to(self._scene)

        self._selected_stream = tk.StringVar()

        self.cam_list: rl.WidgetList
        self.object_list: rl.WidgetList

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

        self._update_cam_table()
        self._update_object_table()

        cams = [c.unique_id for c in self._scene.cameras.values()]
        self.camera_selection.configure(values=cams)
        # listen to all cameras
        for cam in self._scene.cameras.values():
            self.listen_to(cam)

        self.camera_selection.set(cams[-1])
        self._on_camera_selection_change(None)

        objs = [c.name for c in self._scene.objects.values()]
        if len(objs) > 0:
            self.object_selection.configure(values=objs)
            self.object_selection.set(objs[-1])
            self._on_object_selected(None)

        self.t_previous = time.perf_counter()
        self.FPS = 20

        # register task for updating the preview
        self.preview_task = asyncio.get_event_loop().create_task(
            self.live_preview(), name="Live preview"
        )

    def destroy(self) -> None:
        # stop preview task
        logging.info("Cancelling preview task")
        self.preview_task.cancel()
        return super().destroy()

    def update_observer(self, subject, event: Event, *args, **kwargs):
        if event in [
            Event.CAMERA_CALIBRATED,
            Event.CAMERA_ATTACHED,
            Event.CAMERA_ADDED,
            Event.ROBOT_ADDED,
        ]:
            # update calibrated column
            self._update_cam_table()
            if event == Event.CAMERA_ADDED:
                self.listen_to(kwargs["camera"])

            available_cameras = [c.unique_id for c in self._scene.cameras.values()]
            self.camera_selection.configure(values=available_cameras)

        elif event in [Event.OBJECT_REMOVED, Event.OBJECT_REGISTERED, Event.OBJECT_ADDED]:
            self._update_object_table()
            available_objects = [o.name for o in self._scene.objects.values()]
            self.object_selection.configure(values=available_objects)
            if event == Event.OBJECT_ADDED:
                self.listen_to(kwargs["object"])
            elif event == Event.OBJECT_REMOVED:
                self.stop_listening(kwargs["object"])

    def setup_controls(self, master):
        control_frame = tk.Frame(master)
        control_frame.columnconfigure(0, weight=1)

        cam_select_frame = tk.Frame(control_frame)
        camera_selection_label = tk.Label(cam_select_frame, text="Selected Camera")
        camera_selection_label.grid(row=0, column=0, padx=5, sticky=tk.NW)

        self.camera_selection = ttk.Combobox(cam_select_frame, values=[])
        self.camera_selection.set("")
        self.camera_selection.bind("<<ComboboxSelected>>", self._on_camera_selection_change)
        self.camera_selection.grid(row=0, column=1, sticky=tk.NW, padx=5)

        self.stream_selection = ttk.Combobox(
            cam_select_frame, values=["rgb"], textvariable=self._selected_stream
        )
        self.stream_selection.set("rgb")
        self.stream_selection.grid(
            row=0,
            column=2,
            sticky=tk.NW,
            padx=5,
        )

        obj_select_frame = tk.Frame(control_frame)
        self.object_selection = ttk.Combobox(
            obj_select_frame, values=[o.name for o in self._scene.objects.values()]
        )
        self.object_selection.set("")
        self.object_selection.bind("<<ComboboxSelected>>", self._on_object_selected)
        object_selection_label = tk.Label(obj_select_frame, text="Selected Object")
        object_selection_label.grid(row=0, column=0, padx=5, sticky=tk.NW)
        self.object_selection.grid(row=0, column=1, sticky=tk.NW)

        capture_frame = ttk.Frame(control_frame)
        capture_button = ttk.Button(capture_frame, text="Capture Image", command=self._on_capture)
        self.capture_name = ttk.Entry(capture_frame)
        self.capture_name.grid(row=0, column=0)
        capture_button.grid(row=0, column=1)

        self.cam_list = rl.WidgetList(
            control_frame,
            column_names=["Camera", "Calibrated?", "Attached"],
            columns=[tk.Label, tk.Label, ttk.Combobox],
        )
        self.object_list = rl.WidgetList(
            control_frame,
            column_names=["Object", "Color", "Remove"],
            columns=[tk.Label, tk.Label, ttk.Button],
        )

        cam_select_frame.grid(sticky=tk.NSEW, pady=10)
        obj_select_frame.grid(sticky=tk.NSEW, pady=10)
        capture_frame.grid(sticky=tk.NSEW, pady=10)
        self.cam_list.grid(sticky=tk.NSEW, pady=10)
        self.object_list.grid(sticky=tk.NSEW, pady=10)
        return control_frame

    def setup_preview(self, master):
        preview_frame = ttk.Frame(master)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.live_canvas = rl.ResizableImage(preview_frame, bg="#000000")
        self.live_canvas.grid(sticky=tk.NSEW)
        return preview_frame

    async def _on_capture(self):
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
        if cam is None:
            logging.error("No camera selected")
            return
        idx = len(list(data_folder.glob("images/*.png")))

        frame = cam.get_frame(depth_quality=rl.camera.DepthQuality.INFERENCE)
        if cam.robot is not None:
            robot_pose = await cam.robot.pose
            np.savetxt(str(data_folder / f"poses/{idx}.txt"), robot_pose)
        if frame.rgb is not None:
            cv2.imwrite(  # type: ignore
                str(data_folder / f"images/{idx}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)  # type: ignore
            )
        if frame.depth is not None:
            np.save(str(data_folder / f"depth/{idx}.npy"), frame.depth)

    def _update_cam_table(self) -> None:
        for i, cam in enumerate(self._scene.cameras.values()):
            kwargs_list = [
                {"text": f"{cam.name} ({cam.unique_id})"},
                {
                    "text": "Yes" if cam.is_calibrated() else "No",
                    "fg": "green" if cam.is_calibrated() else "red",
                },
                {"values": ["-"] + list(self._scene.robots.keys())},
            ]

            try:
                row_widgets = self.cam_list.rows[i]
                for w, kwargs in zip(row_widgets, kwargs_list):
                    w.configure(**kwargs)
            except IndexError:
                row_widgets = self.cam_list.add_new_row(kwargs_list)

            w_robot: ttk.Combobox = row_widgets[2]  # type: ignore
            w_robot.bind(
                sequence="<<ComboboxSelected>>",
                func=lambda event, cam_unique_id=cam.unique_id, robot=w_robot: self._on_attach_cam(
                    cam_unique_id, robot
                ),
            )

            w_robot.set("-" if cam.robot is None else cam.robot.name)
        # remove the rest of the rows
        for i in range(len(self._scene.cameras), len(self.cam_list.rows)):
            self.cam_list.pop(i)

    def _update_object_table(self) -> None:
        for i, obj in enumerate(self._scene.objects.values()):
            kwargs_list = [
                {"text": obj.name},  # obj_name
                {"bg": "#%02x%02x%02x" % tuple(obj.semantic_color)},  # w_color
                {"text": "x", "command": lambda obj=obj: self._scene.remove_object(obj)},
            ]
            try:
                row_widgets = self.object_list.rows[i]
                for w, kwargs in zip(row_widgets, kwargs_list):
                    w.configure(**kwargs)
            except IndexError:
                row_widgets = self.object_list.add_new_row(kwargs_list)

            _, w_color, _ = row_widgets

            w_color.bind(
                "<Button-1>",
                lambda _: self._on_object_color_click(obj),
            )
        # remove the rest of the rows
        for i in range(len(self._scene.objects), len(self.object_list.rows)):
            self.object_list.pop(i)

    @rl.as_async_task
    async def _on_attach_cam(self, cam_unique_id, w_robot):
        cam = self._scene.cameras[cam_unique_id]

        if w_robot.get() == "-":
            cam.detach()
        else:
            robot = self._scene.robots[w_robot.get()]
            link_mat = cam.extrinsic_matrix if cam.is_calibrated() else np.eye(4)
            await cam.attach(robot, link_mat)

    def _on_camera_selection_change(self, _):
        selected = self.camera_selection.get()
        self._scene.select_camera_by_id(selected)

    def _on_object_selected(self, _):
        selected = self.object_selection.get()
        self._scene.select_object_by_name(selected)

    async def live_preview(self):
        self.t_previous_frame = time.perf_counter()
        self.previous_dts = [1 / self.FPS] * 10
        while True:
            # framerate limiter
            t = time.perf_counter()
            if (t - self.t_previous) < 1 / self.FPS:
                await asyncio.sleep(1 / self.FPS - (t - self.t_previous))
            self.t_previous = time.perf_counter()

            await self.show_single_frame()
            # if we don't update the GUI in this loop, the FPS does not reflect the actual screen FPS
            self.live_canvas.update()

    async def show_single_frame(self):
        selected_cam = self._scene.selected_camera

        if selected_cam is None:
            self.live_canvas.clear_image()
            return

        try:
            # don't change otherwise running a task will trigger constant switching
            frame = selected_cam.get_frame(depth_quality=rl.camera.DepthQuality.UNCHANGED)
        except Exception as e:
            logging.error(f"Failed to get frame from camera: {e}")
            return

        self.stream_selection.configure(
            values=[k for k, v in dataclasses.asdict(frame).items() if v is not None]
        )

        stream_name = self._selected_stream.get()
        try:
            preview = getattr(frame, stream_name)
        except AttributeError:
            logging.debug(f"Stream {stream_name} not available")
            self.live_canvas.clear_image()
            return

        if "depth" in stream_name:
            preview = self._color_depth(preview)

        if "_R" not in stream_name:
            preview = await self._calibrator.draw_on_preview(selected_cam, preview)
            preview = await self._registration.draw_on_preview(selected_cam, preview)

        # draw FPS in top left cornere
        dt = time.perf_counter() - self.t_previous_frame
        self.previous_dts.pop()
        self.previous_dts.insert(0, dt)
        fps = 1 / np.mean(self.previous_dts)

        self.t_previous_frame = time.perf_counter()
        cv2.putText(  # type: ignore
            preview,
            f"{fps:.1f} FPS",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
            2,
            (0, 255, 0),  # draws on RGB image!
            3,
            cv2.LINE_AA,  # type: ignore
        )

        self.live_canvas.set_image(preview)

    def _color_depth(self, depth) -> np.ndarray:
        return cv2.cvtColor(  # type: ignore
            cv2.applyColorMap(  # type: ignore
                cv2.convertScaleAbs(depth, alpha=255 / 2.0), cv2.COLORMAP_JET  # type: ignore
            ),
            cv2.COLOR_BGR2RGB,  # type: ignore
        )

    def _on_object_color_click(self, obj: rl.LabelledObject):
        colors = askcolor(title="Object Semantic Color")
        obj.semantic_color = colors[0]
        self._update_object_table()
