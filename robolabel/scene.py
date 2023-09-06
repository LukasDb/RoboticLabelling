from typing import Literal

import robolabel as rl
from robolabel import Event


class Scene(rl.Observable):
    def __init__(self) -> None:
        rl.Observable.__init__(self)
        self.objects: dict[str, rl.LabelledObject] = {}
        self.robots: dict[str, rl.robot.Robot] = {}
        self.cameras: dict[str, rl.camera.Camera] = {}
        self.background = rl.operators.BackgroundMonitor()
        self.lights = rl.operators.LightsController()
        self.selected_camera: rl.camera.Camera | None = None
        self.selected_object: rl.LabelledObject | None = None

        self.mode: Literal["acquisition", "calibration", "registration"] = "calibration"

    def add_camera(self, camera: rl.camera.Camera):
        self.cameras.update({camera.unique_id: camera})
        self.notify(Event.CAMERA_ADDED, camera=camera)

    def add_robot(self, robot: rl.robot.Robot):
        self.robots.update({robot.name: robot})
        self.notify(Event.ROBOT_ADDED, robot=robot)

    def add_object(self, obj: rl.LabelledObject):
        self.objects.update({obj.name: obj})
        self.notify(Event.OBJECT_ADDED, object=obj)

    def remove_object(self, obj: rl.LabelledObject):
        self.objects.pop(obj.name)
        self.notify(Event.OBJECT_REMOVED, object=obj)

    def select_camera_by_id(self, unique_id: str):
        self.selected_camera = self.cameras[unique_id]
        self.notify(Event.CAMERA_SELECTED, camera=self.selected_camera)

    def select_object_by_name(self, name):
        self.selected_object = self.objects[name]
        self.notify(Event.OBJECT_SELECTED, object=self.selected_object)

    def change_mode(self, mode: Literal["acquisition", "calibration", "registration"]):
        self.mode = mode
        self.notify(Event.MODE_CHANGED, mode=mode)

    def __str__(self) -> str:
        return f"Scene"
