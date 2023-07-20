import open3d as o3d
from pathlib import Path
import numpy as np

from .entity import Entity
from model.observer import Observable, Event


class LabelledObject(Observable, Entity):
    def __init__(self, name, mesh_path: Path):
        Entity.__init__(self, name=name)
        Observable.__init__(self)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path.resolve()))
        self._mesh = mesh

        self.mesh_path = mesh_path
        self.registered = False

        self.semantic_color = [
            hash(self.name) % 255,
            (hash(self.name) - 255) % 255,
            (hash(self.name) - 2 * 255) % 255,
        ]

    def register_pose(self, pose):
        self.registered = True
        self.pose = pose
        self.notify(Event.OBJECT_REGISTERED, object=self)

    @property
    def mesh(self):
        return self._mesh
