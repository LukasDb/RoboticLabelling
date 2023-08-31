import numpy as np
import time
import logging
import asyncio
import tkinter as tk

from robolabel.camera import Camera
from robolabel.robot import Robot
from robolabel.background_monitor import BackgroundMonitor, BackgroundSettings
from robolabel.lights_controller import LightsController, LightsSettings
from .dataset_writer import DatasetWriter
from robolabel.lib.geometry import invert_homogeneous
import itertools as it


class Acquisition:
    def __init__(self) -> None:
        self.acq_thread = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    async def execute(
        self,
        cameras: list[Camera],
        trajectory: list[np.ndarray],
        writer: DatasetWriter | None,
        bg_monitor: BackgroundMonitor,
        bg_settings: BackgroundSettings,
        lights_controller: LightsController,
        lights_settings: LightsSettings,
    ) -> None:
        """executes given trajectory for the selected camera
        1. move to home pose
        2. for each trajectory point:
            for each camera:
                - move to point
                for each light, bg step:  (same step for all cameras!)
                    - take picture
                    - write to dataset
        3. move to home pose
        """
        self.cameras = cameras
        self.trajectory = trajectory
        self.writer = writer
        self.bg_monitor = bg_monitor
        self.bg_settings = bg_settings
        self.lights_controller = lights_controller
        self.lights_settings = lights_settings

        # check that all active cameras belong to the same robot
        if len(cameras) == 0:
            raise ValueError("No cameras selected")

        if len(set([c.robot for c in cameras])) != 1:
            raise ValueError("All selected cameras must be attached to the same robot")

        if cameras[0].robot is None:
            logging.error("Cameras must be attached to a a robot")
            return

        robot: Robot = cameras[0].robot

        if robot.home_pose is None:
            raise ValueError("Robot home pose is not set")

        await robot.move_to(robot.home_pose)

        for pose in trajectory:
            # generate randomized bg and lights settings, to be re-used for all cameras
            bg_steps = bg_monitor.get_steps(bg_settings)
            light_steps = lights_controller.get_steps(lights_settings)

            for cam in cameras:
                robot_target = pose @ invert_homogeneous(cam.extrinsic_matrix)
                await robot.move_to(robot_target)

                for bg_step, light_step in it.product(bg_steps, light_steps):
                    if self._cancelled:
                        self._cancelled = False
                        return
                    bg_monitor.set_step(bg_step)
                    lights_controller.set_step(light_step)

                    print("TAKING IMAGE")
                    await asyncio.sleep(0.5)

                    # TODO write data, pre acq,

        await robot.move_to(robot.home_pose)
