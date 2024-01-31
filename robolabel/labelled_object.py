import open3d as o3d
from pathlib import Path
import numpy as np

import robolabel as rl


class LabelledObject(rl.Observable, rl.Entity):
    def __init__(
        self, name: str, mesh_path: Path, semantic_color: np.ndarray | None = None
    ) -> None:
        rl.Entity.__init__(self, name=name)
        rl.Observable.__init__(self)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path.resolve()))
        self._mesh = mesh

        self.mesh_path = mesh_path

        self.semantic_color: np.ndarray = (
            np.random.randint(0, 255, size=3).tolist()
            if semantic_color is None
            else semantic_color
        )

    @property
    def mesh(self) -> o3d.geometry.TriangleMesh:
        return self._mesh

    def __str__(self) -> str:
        return f"LabelledObject({self.name})"

    def __repr__(self) -> str:
        return f"LabelledObject({self.name})"
