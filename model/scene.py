from typing import List, Dict

from model.observer import Observable, Event
from model.labelled_object import LabelledObject
from model.robot import Robot
from model.camera.camera import Camera
from model.background_monitor import BackgroundMonitor


class Scene(Observable):
    def __init__(self) -> None:
        Observable.__init__(self)
        self.objects: Dict[str, LabelledObject] = {}
        self.robots: Dict[str, Robot] = {}
        self.cameras: Dict[str, Camera] = {}
        self.background = BackgroundMonitor()
        # TODO self.ligthing = LightController()
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

    def select_camera_by_id(self, unique_id):
        self.selected_camera = self.cameras[unique_id]

    def __str__(self) -> str:
        return f"Scene"
