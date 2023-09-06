import open3d as o3d
from pathlib import Path
import numpy as np

import robolabel as rl


class LabelledObject(rl.Observable, rl.Entity):
    def __init__(self, name, mesh_path: Path, semantic_color=None):
        rl.Entity.__init__(self, name=name)
        rl.Observable.__init__(self)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path.resolve()))
        self._mesh = mesh

        self.mesh_path = mesh_path

        if semantic_color is None:
            semantic_color = np.random.randint(0, 255, size=3).tolist()
        self.semantic_color = semantic_color

    def register_pose(self, pose):
        self.pose = pose
        self.notify(rl.Event.OBJECT_REGISTERED, object=self)

    @property
    def mesh(self):
        return self._mesh

    def __str__(self):
        return f"LabelledObject({self.name})"

    def __repr__(self):
        return f"LabelledObject({self.name})"
