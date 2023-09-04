import tkinter as tk
from tkinter import ttk
import logging
import asyncio

from robolabel.scene import Scene
from ..lib.widget_list import WidgetList
from ..lib.settings_widget import SettingsWidget
from robolabel.labelled_object import LabelledObject
from robolabel.camera import Camera
from robolabel.observer import Observable, Observer, Event
from robolabel.operators import (
    TrajectoryGenerator,
    TrajectorySettings,
    DatasetWriter,
    WriterSettings,
    Acquisition,
)
from robolabel.background_monitor import BackgroundSettings
from robolabel.lights_controller import LightsSettings
from contextlib import contextmanager
import copy


def ensure_free_robot(func):
    def wrapper(self, *args, **kwargs):
        # check if 'acquisition' task is running
        if any(t.get_name() == "acquisition" for t in asyncio.all_tasks()):
            logging.error("Acquisition is already running")
            return
        return func(self, *args, **kwargs)

    return wrapper


class ViewAcquisition(tk.Frame, Observer):
    def __init__(self, master, scene: Scene) -> None:
        tk.Frame.__init__(self, master)
        self.scene = scene
        self.listen_to(self.scene)

        self.writer = DatasetWriter()
        self.trajectory_generator = TrajectoryGenerator()
        self.acquisition = Acquisition()

        self.title = ttk.Label(self, text="3. Acquisition")
        self.title.grid(columnspan=3)

        self.cam_list = WidgetList(self, column_names=["Cameras"], columns=[tk.Checkbutton])
        self.object_list = WidgetList(self, column_names=["Objects"], columns=[tk.Checkbutton])
        self.controls = ttk.Frame(self)
        self.settings = ttk.Frame(self)

        padx = 10
        self.cam_list.grid(column=0, row=1, sticky=tk.NSEW, padx=padx)
        self.object_list.grid(column=1, row=1, sticky=tk.NSEW, padx=padx)
        self.controls.grid(column=2, row=1, sticky=tk.NSEW, padx=padx)
        self.settings.grid(column=3, row=1, sticky=tk.NSEW, padx=padx)

        # -- settings --
        self.w_trajectory_settings = SettingsWidget(self.settings, dataclass=TrajectorySettings)
        self.w_background_settings = SettingsWidget(self.settings, dataclass=BackgroundSettings)
        self.w_lights_settings = SettingsWidget(self.settings, dataclass=LightsSettings)
        self.w_writer_settings = SettingsWidget(self.settings, dataclass=WriterSettings)

        self.w_trajectory_settings.grid(column=0, row=1, sticky=tk.NSEW)
        self.w_background_settings.grid(column=0, row=2, sticky=tk.NSEW)
        self.w_lights_settings.grid(column=0, row=3, sticky=tk.NSEW)
        self.w_writer_settings.grid(column=0, row=4, sticky=tk.NSEW)

        self.object_states = {}
        self.camera_states = {}

        # -- controls --
        self.generate_trajectory_button = ttk.Button(
            self.controls,
            text="1. Generate Trajectory",
            command=self._generate_trajectory,
        )
        self.view_trajectory_button = ttk.Button(
            self.controls,
            text="View Trajectory",
            command=lambda: self.trajectory_generator.visualize_trajectory(
                self._active_cameras()[0],
                self._active_objects(),
            ),
        )

        self.dry_run_button = ttk.Button(
            self.controls,
            text="1.2 Perform Dry Run",
            command=self._dry_run,
        )

        self.pre_label = ttk.Label(
            self.controls,
            text="2. Prepare Pre-Acquisition",
        )

        self.start_pre_acquisition = ttk.Button(
            self.controls,
            text="3. Start Pre-Acquisition",
            command=self._start_pre_acquisition,
        )

        self.after_pre_label = ttk.Label(
            self.controls,
            text="4. After Pre-Acquisition",
        )

        self.start_acquisition = ttk.Button(
            self.controls,
            text="5. Start Acquisition",
            command=self._start_acquisition,
        )

        self.cancel_button = ttk.Button(
            self.controls,
            text="Cancel",
            command=lambda: self.acquisition.cancel(),
        )

        pady = 5
        self.generate_trajectory_button.grid(column=0, sticky=tk.W, pady=pady)
        self.view_trajectory_button.grid(column=0, sticky=tk.W, pady=2)
        self.dry_run_button.grid(column=0, sticky=tk.W, pady=pady)
        self.pre_label.grid(column=0, sticky=tk.W, pady=pady)
        self.start_pre_acquisition.grid(column=0, sticky=tk.W, pady=pady)
        self.after_pre_label.grid(column=0, sticky=tk.W, pady=pady)
        self.start_acquisition.grid(column=0, sticky=tk.W, pady=pady)
        self.cancel_button.grid(column=0, sticky=tk.W, pady=pady)

        self._update_cam_table()
        self._update_object_table()

    def _generate_trajectory(self) -> None:
        self.trajectory_generator.generate_trajectory(
            self._active_objects(),
            self.w_trajectory_settings.get_instance(),
        )

    @ensure_free_robot
    def _dry_run(self) -> None:
        with self.overwrite_settings(self.w_writer_settings, {"use_writer": False}):
            self._run_acquisition()

    @ensure_free_robot
    def _start_acquisition(self) -> None:
        with self.overwrite_settings(self.w_writer_settings, {"is_pre_acquisition": False}):
            self._run_acquisition()

    @ensure_free_robot
    def _start_pre_acquisition(self) -> None:
        # TODO change writer to pre-acquisition mode

        self.scene.background.set_textured()

        with (
            self.overwrite_settings(self.w_background_settings, {"use_backgrounds": False}),
            self.overwrite_settings(self.w_lights_settings, {"use_lights": False}),
            self.overwrite_settings(self.w_writer_settings, {"is_pre_acquisition": True}),
        ):
            self._run_acquisition()

    def _run_acquisition(
        self,
    ) -> None:
        logging.info("Starting acquisition...")

        trajectory = self.trajectory_generator.get_current_trajectory()

        bg_monitor = self.scene.background
        lights_controller = self.scene.lights
        writer = self.writer

        asyncio.get_event_loop().create_task(
            self.acquisition.execute(
                self._active_cameras(),
                self._active_objects(),
                trajectory,
                writer=writer,
                writer_settings=self.w_writer_settings.get_instance(),
                bg_monitor=bg_monitor,
                bg_settings=self.w_background_settings.get_instance(),
                lights_controller=lights_controller,
                lights_settings=self.w_lights_settings.get_instance(),
            ),
            name="acquisition",
        )

    def update_observer(self, subject: Observable, event: Event, *args, **kwargs) -> None:
        if event == Event.CAMERA_ADDED:
            self._update_cam_table()
        elif event == Event.OBJECT_ADDED:
            self._update_object_table()

    def _update_cam_table(self) -> None:
        self.camera_states.clear()
        self.cam_list.clear()
        for cam in self.scene.cameras.values():
            cam_state = tk.BooleanVar(value=True)
            self.cam_list.add_new_row(
                [
                    {"text": cam.unique_id, "variable": cam_state},
                ]
            )
            self.camera_states[cam.unique_id] = cam_state

    def _update_object_table(self) -> None:
        self.object_list.clear()
        self.object_states.clear()
        for obj in self.scene.objects.values():
            obj_state = tk.BooleanVar(value=True)
            self.object_list.add_new_row([{"text": obj.name, "variable": obj_state}])
            self.object_states[obj.name] = obj_state

    def _active_cameras(self) -> list[Camera]:
        return [c for c in self.scene.cameras.values() if self.camera_states[c.unique_id].get()]

    def _active_objects(self) -> list[LabelledObject]:
        return [o for o in self.scene.objects.values() if self.object_states[o.name].get()]

    @contextmanager
    def overwrite_settings(self, w_settings: SettingsWidget, overwrite: dict):
        settings = w_settings.get_instance()
        old_settings = copy.deepcopy(settings)

        for k, v in overwrite.items():
            setattr(settings, k, v)

        w_settings.set_from_instance(settings)
        try:
            yield
        finally:
            w_settings.set_from_instance(old_settings)
