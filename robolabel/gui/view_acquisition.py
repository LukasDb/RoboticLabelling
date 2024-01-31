import tkinter as tk
from tkinter import ttk
import logging
import asyncio
from contextlib import contextmanager
import typing
from typing import Callable, Any
import copy

import robolabel as rl

from robolabel.operators import (
    TrajectorySettings,
    BackgroundSettings,
    LightsSettings,
    AcquisitionSettings,
)


P = typing.ParamSpec("P")


def ensure_free_robot(func: Callable[P, None]) -> Callable[P, None]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        # check if 'acquisition' task is running
        if any(t.get_name() == "acquisition" for t in asyncio.all_tasks()):
            logging.error("Acquisition is already running")
            return
        return func(*args, **kwargs)

    return wrapper


class ViewAcquisition(tk.Frame, rl.Observer):
    def __init__(self, master: tk.Misc, scene: rl.Scene) -> None:
        tk.Frame.__init__(self, master)
        self.scene = scene
        self.listen_to(self.scene)

        self.acquisition = rl.operators.DataAcquisition(scene)

        self.title = ttk.Label(self, text="3. Acquisition")
        self.title.grid(columnspan=3)

        self.cam_list = rl.lib.WidgetList(self, column_names=["Cameras"], columns=[tk.Checkbutton])
        self.object_list = rl.lib.WidgetList(
            self, column_names=["Objects"], columns=[tk.Checkbutton]
        )
        self.controls = ttk.Frame(self)
        self.settings = ttk.Frame(self)

        padx = 10
        self.cam_list.grid(column=0, row=1, sticky=tk.NSEW, padx=padx)
        self.object_list.grid(column=1, row=1, sticky=tk.NSEW, padx=padx)
        self.controls.grid(column=2, row=1, sticky=tk.NSEW, padx=padx)
        self.settings.grid(column=3, row=1, sticky=tk.NSEW, padx=padx)

        # -- settings --
        self.w_trajectory_settings = rl.lib.SettingsWidget(
            self.settings, dataclass=TrajectorySettings
        )
        self.w_background_settings = rl.lib.SettingsWidget(
            self.settings, dataclass=BackgroundSettings
        )
        self.w_lights_settings = rl.lib.SettingsWidget(self.settings, dataclass=LightsSettings)
        self.w_writer_settings = rl.lib.SettingsWidget(
            self.settings, dataclass=AcquisitionSettings
        )

        self.w_trajectory_settings.grid(column=0, row=1, sticky=tk.NSEW)
        self.w_background_settings.grid(column=0, row=2, sticky=tk.NSEW)
        self.w_lights_settings.grid(column=0, row=3, sticky=tk.NSEW)
        self.w_writer_settings.grid(column=0, row=4, sticky=tk.NSEW)

        self.object_states: dict[str, tk.BooleanVar] = {}
        self.camera_states: dict[str, tk.BooleanVar] = {}

        # -- controls --
        self.generate_trajectory_button = ttk.Button(
            self.controls,
            text="1. Generate Trajectory",
            command=self._generate_trajectory,
        )
        self.view_trajectory_button = ttk.Button(
            self.controls,
            text="View Trajectory",
            command=self._on_visualize_trajectory,
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
            command=lambda: self.acquisition.trajectory_executor.cancel(),
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

    def _generate_trajectory(self) -> None:
        self.acquisition.trajectory_generator.generate_trajectory(
            self.get_active_objects(),
            self.w_trajectory_settings.get_instance(),
        )

    def _on_visualize_trajectory(self) -> None:
        self.acquisition.trajectory_generator.visualize_trajectory(
            self.get_active_cameras()[0], self.get_active_objects()
        )

    @ensure_free_robot
    def _dry_run(self) -> None:
        with (
            self.overwrite_settings(self.w_writer_settings, {"is_dry_run": True}),
            self.overwrite_settings(self.w_background_settings, {"use_backgrounds": False}),
            self.overwrite_settings(self.w_lights_settings, {"use_lights": False}),
        ):
            self.run_acquisition()

    @ensure_free_robot
    def _start_acquisition(self) -> None:
        with self.overwrite_settings(self.w_writer_settings, {"is_pre_acquisition": False}):
            self.run_acquisition()

    @ensure_free_robot
    def _start_pre_acquisition(self) -> None:
        self.scene.background.set_textured()

        with (
            self.overwrite_settings(self.w_background_settings, {"use_backgrounds": False}),
            self.overwrite_settings(self.w_lights_settings, {"use_lights": False}),
            self.overwrite_settings(self.w_writer_settings, {"is_pre_acquisition": True}),
        ):
            self.run_acquisition()

    def run_acquisition(
        self,
    ) -> None:
        """synchronous wrapper"""
        logging.info("Starting acquisition...")
        acquisition_settings: rl.operators.AcquisitionSettings = (
            self.w_writer_settings.get_instance()
        )
        active_objects = self.get_active_objects()
        active_cameras = self.get_active_cameras()
        bg_settings = self.w_background_settings.get_instance()
        lights_settings = self.w_lights_settings.get_instance()

        asyncio.get_event_loop().create_task(
            self.acquisition.run(
                acquisition_settings=acquisition_settings,
                active_objects=active_objects,
                active_cameras=active_cameras,
                bg_settings=bg_settings,
                lights_settings=lights_settings,
            ),
            name="acquisition",
        )

    def update_observer(
        self, subject: rl.Observable, event: rl.Event, *args: Any, **kwargs: Any
    ) -> None:
        if event == rl.Event.CAMERA_ADDED:
            self._update_cam_table()
        elif event == rl.Event.OBJECT_ADDED:
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

    def get_active_cameras(self) -> list[rl.camera.Camera]:
        return [c for c in self.scene.cameras.values() if self.camera_states[c.unique_id].get()]

    def get_active_objects(self) -> list[rl.LabelledObject]:
        return [o for o in self.scene.objects.values() if self.object_states[o.name].get()]

    def _update_object_table(self) -> None:
        self.object_list.clear()
        self.object_states.clear()
        for obj in self.scene.objects.values():
            obj_state = tk.BooleanVar(value=True)
            self.object_list.add_new_row([{"text": obj.name, "variable": obj_state}])
            self.object_states[obj.name] = obj_state

    @contextmanager
    def overwrite_settings(
        self, w_settings: rl.lib.SettingsWidget, overwrite: dict
    ) -> typing.Generator:
        settings = w_settings.get_instance()
        old_settings = copy.deepcopy(settings)

        for k, v in overwrite.items():
            if not hasattr(settings, k):
                raise ValueError(f"Trying to overwrite unknown setting: {k}")
            setattr(settings, k, v)

        w_settings.set_from_instance(settings)
        try:
            yield
        finally:
            w_settings.set_from_instance(old_settings)
