import open3d as o3d
from typing import List, Tuple
import logging
from robolabel.labelled_object import LabelledObject
import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools as it
from dataclasses import dataclass


@dataclass
class TrajectorySettings:
    n_steps: int = 100  # number of steps in the trajectory
    r_range: Tuple[float, float] = (0.4, 1.0)  # min and max distance from center of objects
    z_cutoff: float = 0.2  # minimum height above the objects
    reach_dist_cutoff: float = 0.4  # how much to reach "behind" the objects
    min_robot_dist: float = 0.3  # minimum distance to the robot
    view_jitter: float = 0.2  # how much to jitter the view direction in m as offset of the center


class TrajectoryGenerator:
    def __init__(self):
        self._current_trajectory = None

    def generate_trajectory(
        self, active_objects: List[LabelledObject], settings: TrajectorySettings
    ):
        """generates a trajectory based on the selected objects.
        The points are generated in a hemisphere above the center of the objects.
        Afterwards the hemisphere is again cut in half towards the robot (is in center of world coordinates)
        """
        if len(active_objects) == 0:
            logging.warning("No objects selected, cannot generate trajectory")
            return

        if self._current_trajectory is not None:
            logging.warning("Overwriting current trajectory")

        object_positions = np.array([o.get_position() for o in active_objects])
        center = np.mean(object_positions, axis=0)

        # random sample on unit sphere
        rots = R.random(2000)
        rots = np.array(rots)

        center_normed = center / np.linalg.norm(center)
        center_dist = np.linalg.norm(center)

        # rejection sampling of points on the sphere
        positions = []
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

            positions.append(pos)

        # downsample the points to the required number of steps
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(positions)
        positions = np.asarray(pcl.farthest_point_down_sample(settings.n_steps).points)

        # resort the positions to always go to the closest one
        arranged_positions = []
        candidates = list(range(len(positions)))
        current_i = np.random.choice(np.arange(len(positions)))
        for _ in range(len(positions)):
            candidates.remove(current_i)
            current_pos = positions[current_i]

            arranged_positions.append(current_pos)

            if candidates == []:
                break

            dists = np.linalg.norm(
                current_pos[np.newaxis, :] - positions[candidates], axis=-1
            )  # [n_points]
            closest = np.argmin(dists)
            current_i = candidates[closest]

        positions = arranged_positions

        # generate poses from positions
        trajectory = []
        for pos in positions:
            view_center = center + np.random.uniform(
                -settings.view_jitter, settings.view_jitter, size=3
            )
            to_point = pos - view_center
            view_yaw = np.arctan2(to_point[1], to_point[0]) + np.pi / 2
            view_pitch = -np.pi / 2 - np.arctan2(to_point[2], np.linalg.norm(to_point[:2]))
            towards_origin = R.from_euler("ZYX", [view_yaw, 0.0, view_pitch])

            random_roll = R.from_euler("Z", np.random.uniform(-np.pi, np.pi)).as_matrix()
            towards_origin = towards_origin.as_matrix() @ random_roll

            pose = np.eye(4)
            pose[:3, 3] = pos
            pose[:3, :3] = towards_origin
            trajectory.append(pose)

        logging.info(f"Generated trajectory with {len(trajectory)} poses")
        self._current_trajectory = trajectory

        # visualize the trajectory for debugging
        poses = []
        for t in trajectory:
            pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pose.compute_vertex_normals()
            pose.transform(t)
            poses.append(pose)

        torus = o3d.geometry.TriangleMesh.create_torus(0.2, 0.05)
        torus.compute_vertex_normals()
        torus.translate(center)

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector([t[:3, 3] for t in trajectory])
        lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(trajectory) - 1)])
        lines.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([1.0, 0.0, 0.0]), (len(trajectory), 1))
        )

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        o3d.visualization.draw_geometries([*poses, torus, lines, origin])

    def get_current_trajectory(self) -> None | List[np.ndarray]:
        if self._current_trajectory is None:
            logging.warning("No trajectory generated yet")
            return None
        return self._current_trajectory
