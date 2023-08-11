import open3d as o3d
from pathlib import Path
import numpy as np

from .entity import Entity
from model.observer import Observable, Event


class LabelledObject(Observable, Entity):
    def __init__(self, name, mesh_path: Path, semantic_color=None):
        Entity.__init__(self, name=name)
        Observable.__init__(self)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path.resolve()))
        self._mesh = mesh

        self.mesh_path = mesh_path
        self.registered = False

        if semantic_color is None:
            semantic_color = np.random.randint(0, 255, size=3).tolist()
        self.semantic_color = semantic_color

    def register_pose(self, pose):
        self.registered = True
        self.pose = pose
        self.notify(Event.OBJECT_REGISTERED, object=self)

    @property
    def mesh(self):
        return self._mesh
