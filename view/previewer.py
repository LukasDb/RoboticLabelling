from scene.scene import Scene
from camera.camera import CamFrame


class Previewer:
    def __init__(self, scene: Scene):
        self._scene = scene

    def draw_scene_onto_rgb(self, frame: CamFrame):
        # draw objects and other scene knowledge onto frame
        return frame.rgb
