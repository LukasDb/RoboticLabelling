from typing import List, Dict

from model.observer import Subject, Event
from model.labelled_object import Labelledobject
from model.robot import Robot
from model.camera.camera import Camera
from model.background_monitor import BackgroundMonitor


class Scene(Subject):
    def __init__(self) -> None:
        Subject.__init__(self)
        self.objects: Dict[str, Labelledobject] = {}
        self.robots: Dict[str, Robot] = {}
        self.cameras: Dict[str, Camera] = {}
        self.background = BackgroundMonitor()
        # TODO self.ligthing = LightController()

    def add_camera(self, camera: Camera):
        self.cameras.update({camera.unique_id: camera})
        self.notify(Event.CAMERA_ADDED, camera=camera)

    def add_robot(self, robot: Robot):
        self.robots.update({robot.name: robot})
        self.notify(Event.ROBOT_ADDED, robot=robot)

    def add_object(self, obj: Labelledobject):
        self.objects.update({obj.name: obj})
        self.notify(Event.OBJECT_ADDED, object=obj)
