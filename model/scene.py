from typing import Dict, List, TYPE_CHECKING

from model.labelled_object import Labelledobject
from model.robot import Robot
from model.camera.camera import Camera


class Scene:
    def __init__(self) -> None:
        self.objects: List[Labelledobject] = []
        self.robots: List[Robot] = []
        self.cameras: List[Camera] = []
        # TODO self.background = BackgroundMonitor()
        # TODO self.ligthing = LightController()

    def add_camera(self, camera: Camera):
        self.cameras.append(camera)

    def add_robot(self, robot: Robot):
        self.robots.append(robot)

    def add_object(self, obj: Labelledobject):
        self.objects.append(obj)


