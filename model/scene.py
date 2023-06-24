from typing import List, Dict

from model.labelled_object import Labelledobject
from model.robot import Robot
from model.camera.camera import Camera
from model.background_monitor import BackgroundMonitor


class Scene:
    def __init__(self) -> None:
        self.objects: List[Labelledobject] = []
        self.robots: List[Robot] = []
        self.cameras: Dict[str, Camera] = {}
        self.background = BackgroundMonitor()
        # TODO self.ligthing = LightController()

    def add_camera(self, camera: Camera):
        self.cameras.update({camera.unique_id: camera})

    def add_robot(self, robot: Robot):
        self.robots.append(robot)

    def add_object(self, obj: Labelledobject):
        self.objects.append(obj)
