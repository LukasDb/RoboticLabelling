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


class Acquisition:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    async def execute(
        self,
        cameras: list[Camera],
        objects: list[LabelledObject],
        trajectory: list[np.ndarray],
        writer: DatasetWriter,
        writer_settings: WriterSettings,
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
        # check that all active cameras belong to the same robot
        if len(cameras) == 0:
            raise ValueError("No cameras selected")

        if len(set([c.robot for c in cameras])) != 1:
            raise ValueError("All selected cameras must be attached to the same robot")

        if cameras[0].robot is None:
            logging.error("Cameras must be attached to a a robot")
            return

        # check all cameras are calibrated
        for cam in cameras:
            if not cam.is_calibrated():
                logging.error(f"Camera {cam.name} is not calibrated")
                return

        # setup writer
        writer.setup(objects, writer_settings)

        robot: Robot = cameras[0].robot

        logging.debug(
            f"Executing trajectory with:\ncameras: {cameras}\nObjects{objects}\n{writer_settings}\n{bg_settings}\n{lights_settings}"
        )

        if robot.home_pose is None:
            raise ValueError("Robot home pose is not set")

        if not await robot.move_to(robot.home_pose, timeout=30):
            logging.error("Failed to move robot to home pose")
            return

        for pose in trajectory:
            # generate randomized bg and lights settings, to be re-used for all cameras
            bg_steps = bg_monitor.get_steps(bg_settings)
            light_steps = lights_controller.get_steps(lights_settings)

            for cam in cameras:
                robot_target = pose @ invert_homogeneous(cam.extrinsic_matrix)
                if not await robot.move_to(robot_target, timeout=10):
                    logging.error("Failed to move robot to target pose")
                    return

                for bg_step, light_step in it.product(bg_steps, light_steps):
                    # TODO this loop blocks the entire program!

                    bg_monitor.set_step(bg_step)
                    lights_controller.set_step(light_step)

                    # wait for background, lights and camera to settle
                    await asyncio.sleep(0.2)
                    if self._cancelled:
                        self._cancelled = False
                        return

                    if writer_settings.use_writer:
                        try:
                            writer.capture(cam)
                        except Exception as e:
                            logging.error(f"Error while capturing data for camera {cam.name}")
                            logging.error(e)
                            import traceback

                            logging.error(traceback.format_exc())
                            return

        if not await robot.move_to(robot.home_pose, timeout=10):
            logging.error("Failed to move robot to home pose")
            return
