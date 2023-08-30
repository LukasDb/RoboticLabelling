import tkinter as tk
from tkinter import ttk
from robolabel.scene import Scene
from .widget_list import WidgetList
from .settings_widget import SettingsWidget
from robolabel.observer import Observable, Observer, Event
from robolabel.operators import TrajectoryGenerator, TrajectorySettings, DatasetWriter, Acquisition


"""
Layout:
1st column: list to choose cameras
2nd column: list to choose objects
3rd column: controls

controls:
    - checkbox: use backgrounds monitor + field to enter path to backgrounds
    - checkbox: use lighting randomization
    - button: generate trajectory: based on selected objects
    - button: start acquisition of HQ depth (with scanning spray)
    - button: start acquisition of Real depth and RGB (after evaporating)
"""


class ViewAcquisition(tk.Frame, Observer):
    def __init__(self, parent, scene: Scene) -> None:
        tk.Frame.__init__(self, parent)
        self.scene = scene
        self.listen_to(self.scene)

        self.traj_gen = TrajectoryGenerator()
        self.acquisition = Acquisition()
        self.writer = DatasetWriter()

        self.title = ttk.Label(self, text="3. Acquisition")
        self.title.grid(columnspan=3)

        self.cam_list = WidgetList(self, column_names=["Cameras"], columns=[tk.Checkbutton])
        self.object_list = WidgetList(self, column_names=["Objects"], columns=[tk.Checkbutton])
        self.controls = ttk.Frame(self)
        self.traj_settings = SettingsWidget(self, dataclass=TrajectorySettings)
        self.traj_settings.set_from_instance(TrajectorySettings())

        padx = 10
        self.cam_list.grid(column=0, row=1, sticky=tk.NSEW, padx=padx)
        self.object_list.grid(column=1, row=1, sticky=tk.NSEW, padx=padx)
        self.controls.grid(column=2, row=1, sticky=tk.NSEW, padx=padx)
        self.traj_settings.grid(column=3, row=1, sticky=tk.NSEW, padx=padx)

        self.object_states = {}
        self.camera_states = {}

        ## setup controls
        self.use_backgrounds = tk.BooleanVar()
        self.use_backgrounds.set(False)
        self.use_backgrounds_checkbox = tk.Checkbutton(
            self.controls,
            text="Use Backgrounds",
            variable=self.use_backgrounds,
        )

        self.use_lighting = tk.BooleanVar()
        self.use_lighting.set(False)
        self.use_lighting_checkbox = tk.Checkbutton(
            self.controls,
            text="Use Lighting",
            variable=self.use_lighting,
        )

        self.generate_trajectory_button = ttk.Button(
            self.controls,
            text="1. Generate Trajectory",
            command=self._generate_trajectory,
        )
        self.dry_run_button = ttk.Button(
            self.controls,
            text="1.2 Perform Dry Run",
            command=self._dry_run,
        )

        self.pre_label = ttk.Label(
            self.controls,
            text="2. Prepare Pre-Acquisition: Spray the scene with scanning spray.",
        )

        self.start_pre_acquisition = ttk.Button(
            self.controls,
            text="3. Start Pre-Acquisition",
            command=self._start_pre_acquisition,
        )

        self.after_pre_label = ttk.Label(
            self.controls,
            text="4. After Pre-Acquisition: Wait for evaporation of the scanning spray.",
        )

        self.start_acquisition = ttk.Button(
            self.controls,
            text="5. Start Acquisition",
            command=self._start_acquisition,
        )

        pady = 5
        self.use_backgrounds_checkbox.grid(column=0, sticky=tk.W, pady=2)
        self.use_lighting_checkbox.grid(column=0, sticky=tk.W, pady=pady)
        self.generate_trajectory_button.grid(column=0, sticky=tk.W, pady=pady)
        self.dry_run_button.grid(column=0, sticky=tk.W, pady=pady)
        self.pre_label.grid(column=0, sticky=tk.W, pady=pady)
        self.start_pre_acquisition.grid(column=0, sticky=tk.W, pady=pady)
        self.after_pre_label.grid(column=0, sticky=tk.W, pady=pady)
        self.start_acquisition.grid(column=0, sticky=tk.W, pady=pady)

    def _generate_trajectory(self):
        self.traj_gen.generate_trajectory(
            [o for o in self.scene.objects.values() if self.object_states[o.name].get()],
            self.traj_settings.get_instance(),
        )

    def _dry_run(self):
        print("Performing dry run...")
        print(self.traj_settings.get_instance())
        traj = self.traj_gen.get_current_trajectory()
        if traj is None:
            print("No trajectory generated yet")
            return

    def _start_acquisition(self):
        print(
            f"Starting acquisition with bg: {self.use_backgrounds.get()} and lights: {self.use_lighting.get()}"
        )
        print("Selected cameras:")
        for cam_id, state in self.camera_states.items():
            if state.get():
                print(f"\t{cam_id}")
        print("Selected objects:")
        for obj_name, state in self.object_states.items():
            if state.get():
                print(f"\t{obj_name}")

    def _start_pre_acquisition(self):
        print("Starting pre-acquisition...")

    def update_observer(self, subject: Observable, event: Event, *args, **kwargs):
        if event == Event.CAMERA_ADDED:
            self._update_cam_table()
        elif event == Event.OBJECT_ADDED:
            self._update_object_table()

    def _update_cam_table(self):
        self.camera_states.clear()
        self.cam_list.clear()
        for cam in self.scene.cameras.values():
            cam_state = tk.BooleanVar(value="1")
            self.cam_list.add_new_row(
                [
                    {"text": cam.unique_id, "variable": cam_state},
                ]
            )
            self.camera_states[cam.unique_id] = cam_state

    def _update_object_table(self):
        self.object_list.clear()
        self.object_states.clear()
        for obj in self.scene.objects.values():
            obj_state = tk.BooleanVar(value="1")
            self.object_list.add_new_row([{"text": obj.name, "variable": obj_state}])
            self.object_states[obj.name] = obj_state
