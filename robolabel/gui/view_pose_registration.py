import asyncio
from typing import Any
import tkinter as tk
from tkinter import ttk
import logging

import robolabel as rl
from robolabel import Event
from robolabel.geometry import get_euler_from_affine_matrix


class ViewPoseRegistration(rl.Observer, ttk.Frame):
    def __init__(
        self, master: tk.Misc, scene: rl.Scene, registration: rl.operators.PoseRegistration
    ) -> None:
        ttk.Frame.__init__(self, master)
        self.scene = scene
        self.listen_to(self.scene)
        self.registration = registration

        self.title = ttk.Label(self, text="2. Pose Registration")
        self.title.grid(columnspan=2)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, minsize=300, weight=1)
        self.rowconfigure(1, weight=1)

        controls = self.setup_controls(self)
        self.preview = rl.lib.ResizableImage(self, bg="#000000")

        controls.grid(row=1, column=1, sticky=tk.NSEW)
        self.preview.grid(row=1, column=0, sticky=tk.NSEW)

    def update_observer(
        self, subject: rl.Observable, event: Event, *args: Any, **kwargs: Any
    ) -> None:
        if event == Event.OBJECT_ADDED:
            # configure choices for object selection
            self.listen_to(kwargs["object"])
            self._update_gui()

        if event == Event.OBJECT_REMOVED:
            # configure choices for object selection
            self.stop_listening(kwargs["object"])
            self._update_gui()

    def setup_controls(self, master: tk.Misc) -> ttk.Frame:
        control_frame = ttk.Frame(master)

        self.capture_button = ttk.Button(
            control_frame, text="Capture Image", command=self._on_capture
        )

        self.auto_capture_button = ttk.Button(
            control_frame,
            text="Auto Capture",
            command=self._on_automatic_acquisition,
        )
        self.cancel_button = ttk.Button(
            control_frame,
            text="Cancel",
            command=lambda: self.registration.acquisition.cancel(),
        )

        self.image_selection = ttk.Combobox(control_frame)
        self.image_selection.bind(
            sequence="<<ComboboxSelected>>", func=lambda _: self._update_gui()
        )

        # add button
        self.optimize_button = ttk.Button(
            control_frame,
            text="Optimize",
            command=self._on_optimize,
        )
        self.reset_pose_button = ttk.Button(
            control_frame,
            text="Reset Pose",
            command=self._on_reset_pose,
        )
        self.reset_button = ttk.Button(
            control_frame,
            text="Reset",
            command=self._on_reset,
        )

        # add button to move object pose
        pose_frame = ttk.Frame(control_frame)  # 2 columns for pos and orn
        self.position_label = ttk.Label(pose_frame, text="Position:")
        self.orientation_label = ttk.Label(pose_frame, text="Orientation:")
        self.position_label.grid(row=0, column=0, sticky=tk.W)
        self.orientation_label.grid(row=0, column=1, sticky=tk.W)

        def sb_pos(row: int) -> ttk.Spinbox:
            spinbox = ttk.Spinbox(
                pose_frame,
                from_=-1.0,
                to=1.0,
                increment=0.01,
                command=lambda: self._change_initial_guess(),
            )
            # bind self._on_change to spinbox
            spinbox.bind("<Return>", lambda _: self._change_initial_guess())
            spinbox.grid(pady=5, column=0, row=row)
            return spinbox

        def sb_rot(row: int) -> ttk.Spinbox:
            # seperat spinbox for rotation
            spinbox = ttk.Spinbox(
                pose_frame,
                from_=-180,
                to=180,
                increment=10,
                command=lambda: self._change_initial_guess(),
            )
            spinbox.bind("<Return>", lambda _: self._change_initial_guess())
            spinbox.grid(pady=5, column=1, row=row)
            return spinbox

        self.manual_pose_x = sb_pos(1)  # boxes for position user input
        self.manual_pose_y = sb_pos(2)
        self.manual_pose_z = sb_pos(3)

        self.manual_pose_rho = sb_rot(1)  # for angles
        self.manual_pose_phi = sb_rot(2)
        self.manual_pose_theta = sb_rot(3)

        pady = 5

        # control frame
        self.capture_button.grid(pady=pady, row=0, columnspan=2)

        self.auto_capture_button.grid(pady=pady, row=1, column=0)
        self.cancel_button.grid(pady=pady, row=1, column=1)

        self.image_selection.grid(pady=pady, row=2, column=0)
        self.reset_button.grid(pady=pady, row=2, column=1)

        pose_frame.grid(pady=pady, row=3, columnspan=2)

        self.optimize_button.grid(pady=pady, row=4, column=0)
        self.reset_pose_button.grid(pady=pady, row=4, column=1)

        return control_frame

    def _change_initial_guess(self) -> None:
        if self.scene.selected_object is not None:
            (rho, phi, theta), (x, y, z) = get_euler_from_affine_matrix(
                self.scene.selected_object.get_pose()
            )

            if self.manual_pose_x.get():
                x = self.manual_pose_x.get()  # user inputs in spinbox
            if self.manual_pose_y.get():
                y = self.manual_pose_y.get()
            if self.manual_pose_z.get():
                z = self.manual_pose_z.get()
            if self.manual_pose_x.get():
                rho = self.manual_pose_rho.get()  # orientation values
            if self.manual_pose_y.get():
                phi = self.manual_pose_phi.get()
            if self.manual_pose_z.get():
                theta = self.manual_pose_theta.get()

            self.registration.move_pose(self.scene.selected_object, x, y, z, rho, phi, theta)

            self._update_gui()  # paint new mesh

    def _on_capture(self) -> None:
        async def capture_and_update() -> None:
            await self.registration.capture()
            self._update_gui(set_to_last_image=True)

        asyncio.get_event_loop().create_task(
            capture_and_update(),
            name="capture_image",
        )

    def _on_automatic_acquisition(self) -> None:
        # check if asyncio task is already running
        if "auto_registration" in [t.get_name() for t in asyncio.all_tasks()]:
            logging.warn("Auto auto_registration already running!")
            return

        asyncio.get_event_loop().create_task(
            self.registration.capture_images(lambda: self._update_gui(set_to_last_image=True)),
            name="auto_registration",
        )

    def _on_reset_pose(self) -> None:
        monitor_pose = self.scene.background.get_pose()
        if self.scene.selected_object is not None:
            self.scene.selected_object.set_pose(monitor_pose)  # add this as initial position
        self._update_gui()

    def _on_reset(self) -> None:
        self.registration.reset()
        self._update_gui()

    def _on_optimize(self) -> None:
        self.registration.optimize_pose()

    def _update_gui(self, set_to_last_image: bool = False) -> None:
        if self.scene.selected_object is None:
            return

        (rho, phi, theta), (x, y, z) = get_euler_from_affine_matrix(
            self.scene.selected_object.get_pose()
        )

        self.manual_pose_x.set(float(x))  # set first spinbox values to current pose
        self.manual_pose_y.set(float(y))
        self.manual_pose_z.set(float(z))
        self.manual_pose_rho.set(float(rho))
        self.manual_pose_phi.set(float(phi))
        self.manual_pose_theta.set(float(theta))

        # update preview
        if len(self.registration.datapoints) == 0:
            self.preview.clear_image()
            return

        self.image_selection["values"] = [
            f"Image {i:2}" for i in range(len(self.registration.datapoints))
        ]
        if set_to_last_image:
            self.image_selection.set(self.image_selection["values"][-1])

        selected = self.image_selection.get()
        try:
            selected_index = int(selected.split(" ")[-1])
        except ValueError:
            return
        img = self.registration.get_from_image_cache(selected_index)

        if img is not None:
            self.preview.set_image(img)
        else:
            self.preview.clear_image()
