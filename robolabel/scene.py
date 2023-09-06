from typing import Literal

from .observer import Observable, Event
from .labelled_object import LabelledObject
from .robot.robot import Robot
from .camera.camera import Camera
from .background_monitor import BackgroundMonitor
from .lights_controller import LightsController


class Scene(Observable):
    def __init__(self) -> None:
        Observable.__init__(self)
        self.objects: dict[str, LabelledObject] = {}
        self.robots: dict[str, Robot] = {}
        self.cameras: dict[str, Camera] = {}
        self.background = BackgroundMonitor()
        self.lights = LightsController()
        self.selected_camera: Camera | None = None
        self.selected_object: LabelledObject | None = None

        self.mode: Literal["acquisition", "calibration", "registration"] = "calibration"

    def add_camera(self, camera: Camera):
        self.cameras.update({camera.unique_id: camera})
        self.notify(Event.CAMERA_ADDED, camera=camera)

    def add_robot(self, robot: Robot):
        self.robots.update({robot.name: robot})
        self.notify(Event.ROBOT_ADDED, robot=robot)

    def add_object(self, obj: LabelledObject):
        self.objects.update({obj.name: obj})
        self.notify(Event.OBJECT_ADDED, object=obj)

    def remove_object(self, obj: LabelledObject):
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
