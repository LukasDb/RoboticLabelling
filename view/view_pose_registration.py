import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation as R
import itertools as it
from typing import List
from lib.geometry import get_rvec_tvec_from_affine_matrix, get_euler_from_affine_matrix

from model.scene import Scene
from model.observer import Event, Observable, Observer
from view.resizable_image import ResizableImage
from control.pose_registrator import PoseRegistrator, Datapoint


class ViewPoseRegistration(Observer, ttk.Frame):
    def __init__(self, parent, scene: Scene, registrator: PoseRegistrator) -> None:
        ttk.Frame.__init__(self, parent)
        self.scene = scene
        self.listen_to(self.scene)
        self.registrator = registrator

        self.title = ttk.Label(self, text="2. Pose Registration")
        self.title.grid()

        controls = self.setup_controls(self)
        previews = self.setup_previews(self)

        controls.grid(row=1, column=1, sticky=tk.NSEW)
        previews.grid(row=1, column=0, sticky=tk.NSEW)

    def update_observer(self, subject: Observable, event: Event, *args, **kwargs):
        if event == Event.OBJECT_ADDED:
            # configure choices for object selection
            self.object_selection.configure(values=[o.name for o in self.scene.objects.values()])
            self.listen_to(kwargs["object"])
        elif event == Event.OBJECT_REGISTERED:
            # update controls rerender buffer
            self._update_gui_from_object_pose()
            self._preview_buffer()

    def setup_controls(self, parent):
        control_frame = ttk.Frame(parent)

        self.object_selection = ttk.Combobox(
            control_frame, values=[o.name for o in self.scene.objects.values()]
        )
        self.object_selection.bind("<<ComboboxSelected>>", lambda _: self._on_object_selected())

        self.capture_button = ttk.Button(
            control_frame, text="Capture Image", command=self._on_capture
        )

        # add button
        self.update_button = ttk.Button(
            control_frame,
            text="Optimize",
            command=self.registrator.optimize_pose,
        )
        self.reset_pose_button = ttk.Button(
            control_frame,
            text="Reset Pose",
            command=self._on_reset,
        )

        # add button to move object pose
        self.position_label = ttk.Label(control_frame, text="Position:")
        self.orientation_label = ttk.Label(control_frame, text="Orientation:")

        def sb_pos():
            spinbox = ttk.Spinbox(
                control_frame,
                from_=-1.0,
                to=1.0,
                increment=0.05,
                command=lambda: self._change_initial_guess(),
            )
            # bind self._on_change to spinbox
            spinbox.bind("<Return>", lambda _: self._change_initial_guess())
            return spinbox

        def sb_rot():
            # seperat spinbox for rotation
            spinbox = ttk.Spinbox(
                control_frame,
                from_=-180,
                to=180,
                increment=5,
                command=lambda: self._change_initial_guess(),
            )
            spinbox.bind("<Return>", lambda _: self._change_initial_guess())
            return spinbox

        self.manual_pose_x = sb_pos()  # boxes for position user input
        self.manual_pose_y = sb_pos()
        self.manual_pose_z = sb_pos()

        self.manual_pose_rho = sb_rot()  # for angles
        self.manual_pose_phi = sb_rot()
        self.manual_pose_theta = sb_rot()

        pady = 5
        pady_L = 20
        self.capture_button.grid(pady=pady)
        self.object_selection.grid(pady=pady)
        self.position_label.grid(pady=pady_L)
        self.manual_pose_x.grid(pady=pady)
        self.manual_pose_y.grid(pady=pady)
        self.manual_pose_z.grid(pady=pady)
        self.orientation_label.grid(pady=pady_L)
        self.manual_pose_rho.grid(pady=pady)
        self.manual_pose_phi.grid(pady=pady)
        self.manual_pose_theta.grid(pady=pady)
        self.update_button.grid(pady=pady_L)
        self.reset_pose_button.grid(pady=pady)

        return control_frame

    def setup_previews(self, parent):
        preview_frame = ttk.Frame(parent)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        prev0 = ResizableImage(preview_frame, bg="#000000")
        prev0.grid(row=0, column=0, sticky=tk.NSEW)

        prev1 = ResizableImage(preview_frame, bg="#000000")
        prev1.grid(row=0, column=1, sticky=tk.NSEW)

        prev2 = ResizableImage(preview_frame, bg="#000000")
        prev2.grid(row=1, column=0, sticky=tk.NSEW)

        prev3 = ResizableImage(preview_frame, bg="#000000")
        prev3.grid(row=1, column=1, sticky=tk.NSEW)

        self.previews = [prev0, prev1, prev2, prev3]
        return preview_frame

    def _on_object_selected(self):
        self.scene.select_object_by_name(self.object_selection.get())
        self._preview_buffer()
        self._update_gui_from_object_pose()

    def _update_gui_from_object_pose(self):
        (rho, phi, theta), (x, y, z) = get_euler_from_affine_matrix(
            self.scene.selected_object.pose
        )


        self.manual_pose_x.set(float(x))  # set first spinbox values to current pose
        self.manual_pose_y.set(float(y))
        self.manual_pose_z.set(float(z))
        self.manual_pose_rho.set(float(rho))
        self.manual_pose_phi.set(float(phi))
        self.manual_pose_theta.set(float(theta))

    def _change_initial_guess(self):
        if self.scene.selected_object is not None:
            (rho, phi, theta), (x, y, z) = get_euler_from_affine_matrix(
                self.scene.selected_object.pose
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

            self.registrator.move_pose(self.scene.selected_object, x, y, z, rho, phi, theta)

            self._preview_buffer()  # paint new mesh

    def _on_capture(self):
        self.registrator.capture_image()
        self._preview_buffer()

    def _preview_buffer(self):
        for datapoint, preview in zip(self.registrator.datapoints[-4:], self.previews):
            if datapoint is None:
                continue

            img = datapoint.rgb.copy()
            if self.scene.selected_object is not None:
                img = self.registrator.draw_registered_object(
                    self.scene.selected_object,
                    img,
                    datapoint.pose,
                    datapoint.intrinsics,
                    datapoint.dist_coeffs,
                )
            preview.set_image(img)

    def _on_reset(self):
        monitor_pose = self.scene.background.pose
        self.scene.selected_object.pose = monitor_pose  # add this as inital position
        self._update_gui_from_object_pose()
        self._preview_buffer()


# good guess:
# x :  0.42
# y:  -0.58
# z   -0.11
# euler: [90, 0, 180]
