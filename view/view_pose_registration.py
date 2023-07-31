import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation as R
import itertools as it
from typing import List
from lib.geometry import get_rvec_tvec_from_affine_matrix

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
            self.object_selection.configure(
                values=[o.name for o in self.scene.objects.values()]
            )

    def setup_controls(self, parent):
        control_frame = ttk.Frame(parent)

        self.object_selection = ttk.Combobox(
            control_frame, values=[o.name for o in self.scene.objects.values()]
        )
        self.object_selection.bind(
            "<<ComboboxSelected>>", lambda _: self._on_object_selected()
        )

        self.capture_button = ttk.Button(
            control_frame, text="Capture Image", command=self._on_capture
        )

        # add button
        self.update_button = ttk.Button(
            control_frame,
            text="Optimize",
            command=self.registrator.optimize_pose,
        )

        # add button to move object pose
        self.position_label = ttk.Label(
            control_frame, 
            text="Position:"
        )
        
        def sb(var):
            return ttk.Spinbox(
            control_frame,
            from_=-4.0,
            to=1.0,
            increment=0.1,
            command= lambda : self._change_initial_guess(),
            textvariable=var
            )
        xtk=tk.DoubleVar(value=0.0)
        ytk=tk.DoubleVar(value=0.0)
        ztk=tk.DoubleVar(value=0.0)
        self.manual_pose_x = sb(xtk)
        self.manual_pose_y = sb(ytk)
        self.manual_pose_z = sb(ztk)


        pady = 5
        pady_L = 20
        self.capture_button.grid(pady=pady)
        self.object_selection.grid(pady=pady)
        self.position_label.grid(pady=pady_L)
        self.manual_pose_x.grid(pady=pady)
        self.manual_pose_y.grid(pady=pady)
        self.manual_pose_z.grid(pady=pady)
        self.update_button.grid(pady=pady_L)

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

        (x,y,z),_ = get_rvec_tvec_from_affine_matrix(
            self.scene.selected_object.pose)    
        self.manual_pose_x.set(0.0)    #x[0])              #set first spinbox values to current pose
        self.manual_pose_y.set(0.0)    #y[0])
        self.manual_pose_z.set(0.0)    #z[0])
    
    def _change_initial_guess(self):
        if self.scene.selected_object is not None:
            (x,y,z),(rho,phi,theta)=get_rvec_tvec_from_affine_matrix(
                self.scene.selected_object.pose)
            if self.manual_pose_x.get():
                x = self.manual_pose_x.get().replace('\U00002013', '-')  # user inputs in spinbox
            if self.manual_pose_y.get():
                y = self.manual_pose_y.get()
            if self.manual_pose_z.get():
                z = self.manual_pose_z.get()
            self.registrator.move_pose(
                self.scene.selected_object,
                x[0],y[0],z[0]
                )
            self._preview_buffer()      # paint new mesh

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
