from .entity import Entity
import open3d as o3d


class Labelledobject(Entity):
    def __init__(self, mesh):
        super().__init__()
        self._mesh = mesh

    @staticmethod
    def from_obj(obj_path: str):
        mesh = o3d.io.read_triangle_mesh(obj_path)
        return Labelledobject(mesh)
