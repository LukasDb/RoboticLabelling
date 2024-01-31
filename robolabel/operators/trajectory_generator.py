import open3d as o3d
import logging
from robolabel.labelled_object import LabelledObject
from robolabel.camera import Camera
from robolabel.geometry import invert_homogeneous
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class TrajectorySettings:
    n_steps: int = 20  # number of steps in the trajectory
    r_range: tuple[float, float] = (0.3, 0.6)  # min and max distance from center of objects
    z_cutoff: float = 0.4  # minimum height above the objects
    reach_dist_cutoff: float = 0.3  # how much to reach "behind" the objects
    min_robot_dist: float = 0.5  # minimum distance to the robot
    view_jitter: float = 0.05  # how much to jitter the view direction in m as offset of the center
    roll_range: tuple[float, float] = (-180, 180)


class TrajectoryGenerator:
    def __init__(self) -> None:
        self._current_trajectory: None | list[np.ndarray] = None

    def generate_trajectory(
        self, active_objects: list[LabelledObject], settings: TrajectorySettings
    ) -> list[np.ndarray] | None:
        """generates a trajectory based on the selected objects.
        The points are generated in a hemisphere above the center of the objects.
        Afterwards the hemisphere is again cut in half towards the robot (is in center of world coordinates)
        """
        if len(active_objects) == 0:
            logging.warning("No objects selected, cannot generate trajectory")
            return []

        if self._current_trajectory is not None:
            logging.warning("Overwriting current trajectory")

        object_positions = np.array([o.get_position() for o in active_objects])
        center = np.mean(object_positions, axis=0)
        return self.generate_trajectory_above_center(center, settings)

    def generate_trajectory_above_center(
        self, center: np.ndarray, settings: TrajectorySettings
    ) -> list[np.ndarray]:
        # random sample on unit sphere
        rots = R.random(2000)
        rots = np.array(rots)

        center_dist = np.linalg.norm(center)
        center_normed = center / center_dist

        # rejection sampling of points on the sphere
        sphere_points = []
        for rot in rots:
            r = (
                np.sqrt(np.random.uniform(0.0, 1.0)) * (settings.r_range[1] - settings.r_range[0])
                + settings.r_range[0]
            )
            pos = center + rot.as_matrix() @ np.array([0.0, 0.0, r])

            dist = np.linalg.norm(pos)
            θ = np.arccos(np.dot(pos, center_normed) / dist)
            if dist * np.cos(θ) - center_dist > settings.reach_dist_cutoff:
                continue

            if pos[2] < (center[2] + settings.z_cutoff):
                continue

            if np.linalg.norm(pos[:2]) < settings.min_robot_dist:
                continue

            sphere_points.append(pos)

        # downsample the points to the required number of steps
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(sphere_points)
        positions = np.asarray(pcl.farthest_point_down_sample(settings.n_steps).points)

        # resort the positions to always go to the closest one
        sorted_positions = []
        candidates = list(range(len(positions)))

        # current_i = np.random.choice(np.arange(len(positions)))
        # set start highest point
        current_i = int(np.argmax(positions[:, 2]))

        for _ in range(len(positions)):
            candidates.remove(current_i)
            current_pos = positions[current_i]

            sorted_positions.append(current_pos)

            if candidates == []:
                break

            dists = np.linalg.norm(
                current_pos[np.newaxis, :] - positions[candidates], axis=-1
            )  # [n_points]
            closest = np.argmin(dists)
            current_i = candidates[closest]

        # generate poses from positions
        trajectory = []
        for pos in sorted_positions:
            # jitter applies only to the view direction (orientation)
            if isinstance(settings.view_jitter, float):
                view_center = center + np.random.uniform(
                    -settings.view_jitter, settings.view_jitter, size=3
                )
            else:
                view_center = center + np.array(
                    [
                        np.random.uniform(-settings.view_jitter[i], settings.view_jitter[i])
                        for i in range(3)
                    ]
                )

            to_point = pos - view_center
            view_yaw = np.arctan2(to_point[1], to_point[0]) + np.pi / 2
            view_pitch = -np.pi / 2 - np.arctan2(to_point[2], np.linalg.norm(to_point[:2]))
            towards_origin = R.from_euler("ZYX", [view_yaw, 0.0, view_pitch])

            random_roll = R.from_euler(
                "Z", np.random.uniform(*settings.roll_range), degrees=True
            ).as_matrix()
            towards_origin = towards_origin.as_matrix() @ random_roll

            pose = np.eye(4)
            pose[:3, 3] = pos
            pose[:3, :3] = towards_origin
            trajectory.append(pose)

        logging.info(f"Generated trajectory with {len(trajectory)} poses")
        self._current_trajectory = trajectory
        return trajectory

    def visualize_trajectory(self, camera: Camera, objects: list[LabelledObject]) -> None:
        assert self._current_trajectory is not None, "Generate trajectory first"

        w, h = camera.width, camera.height
        try:
            intrinsics = camera.intrinsic_matrix
        except AssertionError:
            intrinsics = np.array([[w, 0.0, w / 2], [0.0, w, h / 2], [0.0, 0.0, 1.0]])

        vis_views = []
        for pose in self._current_trajectory:
            # vis_view = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # generate a camera frustum
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                w,
                h,
                intrinsics,
                invert_homogeneous(pose),
                0.05,
            )

            vis_views.append(frustum)

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector([t[:3, 3] for t in self._current_trajectory])
        lines.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(self._current_trajectory) - 1)]
        )
        lines.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([1.0, 0.0, 0.0]), (len(self._current_trajectory), 1))
        )

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        object_meshes = []
        for obj in objects:
            mesh = o3d.geometry.TriangleMesh(obj.mesh)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(np.array(obj.semantic_color) / 255.0)
            mesh.transform(obj.get_pose())
            object_meshes.append(mesh)

        o3d.visualization.draw_geometries([*vis_views, lines, origin, *object_meshes])  # type: ignore

    def get_current_trajectory(self) -> list[np.ndarray]:
        assert self._current_trajectory is not None, "No trajectory generated yet"
        return self._current_trajectory
