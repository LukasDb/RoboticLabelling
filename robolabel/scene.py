from typing import List, Dict

from .observer import Observable, Event
from .labelled_object import LabelledObject
from .robot.robot import Robot
from .camera.camera import Camera
from .background_monitor import BackgroundMonitor
from .lights_controller import LightsController


class Scene(Observable):
    def __init__(self) -> None:
        Observable.__init__(self)
        self.objects: Dict[str, LabelledObject] = {}
        self.robots: Dict[str, Robot] = {}
        self.cameras: Dict[str, Camera] = {}
        self.background = BackgroundMonitor()
        self.lights = LightsController()
        # TODO self.lighting = LightController()
        self.selected_camera: Camera = None
        self.selected_object: LabelledObject = None

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

    def select_camera_by_id(self, unique_id):
        self.selected_camera = self.cameras[unique_id]

    def select_object_by_name(self, name):
        self.selected_object = self.objects[name]

    def __str__(self) -> str:
        return f"Scene"
