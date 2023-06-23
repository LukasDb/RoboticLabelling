from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
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
