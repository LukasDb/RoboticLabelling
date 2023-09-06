import numpy as np
import time
import logging
import asyncio
import tkinter as tk

from robolabel.camera import Camera
from robolabel.robot import Robot
from robolabel.labelled_object import LabelledObject
from robolabel.background_monitor import BackgroundMonitor, BackgroundSettings
from robolabel.lights_controller import LightsController, LightsSettings
from .dataset_writer import DatasetWriter, WriterSettings
from robolabel.lib.geometry import invert_homogeneous
import itertools as it
import asyncio


class Acquisition:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    async def execute(
        self,
        cameras: list[Camera],
        trajectory: list[np.ndarray],
        bg_monitor: BackgroundMonitor | None = None,
        bg_settings: BackgroundSettings | None = None,
        lights_controller: LightsController | None = None,
        lights_settings: LightsSettings | None = None,
    ):
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
        # check that all active cameras belong to the same robot
        if len(cameras) == 0:
            raise ValueError("No cameras selected")

        if len(set([c.robot for c in cameras])) != 1:
            raise ValueError("All selected cameras must be attached to the same robot")

        if cameras[0].robot is None:
            logging.error("Cameras must be attached to a a robot")
            return

        robot: Robot = cameras[0].robot

        logging.debug(
            f"Executing trajectory with:\n\t- {cameras}\n\t- {bg_settings}\n\t- {lights_settings}"
        )

        if robot.home_pose is None:
            raise ValueError("Robot home pose is not set")

        if not await robot.move_to(robot.home_pose, timeout=50):
            logging.error("Failed to move robot to home pose")
            return
        logging.info("Reached home pose")

        for idx_trajectory, pose in enumerate(trajectory):
            # generate randomized bg and lights settings, to be re-used for all cameras

            bg_steps = [None] if bg_monitor is None else bg_monitor.get_steps(bg_settings)
            light_steps = (
                [None]
                if lights_controller is None
                else lights_controller.get_steps(lights_settings)
            )

            for cam in cameras:
                try:
                    robot_target = pose @ invert_homogeneous(cam.extrinsic_matrix)
                except AssertionError:
                    robot_target = pose
                if not await robot.move_to(robot_target, timeout=30):
                    logging.error("Failed to move robot to target pose")
                    return

                for bg_step, light_step in it.product(bg_steps, light_steps):
                    if bg_monitor is not None:
                        bg_monitor.set_step(bg_step)
                    if lights_controller is not None:
                        lights_controller.set_step(light_step)

                    # wait for background, lights and camera to settle
                    await asyncio.sleep(0.1)
                    if self._cancelled:
                        self._cancelled = False
                        return

                    yield idx_trajectory, cam

        if not await robot.move_to(robot.home_pose, timeout=50):
            logging.error("Failed to move robot to home pose")
            return
