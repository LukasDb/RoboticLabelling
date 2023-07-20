from .entity import Entity
import open3d as o3d
from pathlib import Path


class LabelledObject(Entity):
    def __init__(self, name, mesh_path: Path):
        super().__init__(name=name)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path.resolve()))
        self.mesh_path = mesh_path
        self._mesh = mesh
