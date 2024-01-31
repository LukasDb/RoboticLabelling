import numpy as np
import time
import logging
import asyncio
import tkinter as tk
import itertools as it
import asyncio
from typing import AsyncGenerator

import robolabel as rl
from robolabel.operators import (
    BackgroundMonitor,
    BackgroundSettings,
    LightsController,
    LightsSettings,
)
from robolabel.geometry import invert_homogeneous


class TrajectoryExecutor:
    def __init__(self, scene: rl.Scene) -> None:
        self._cancelled = False
        self._scene = scene
        self.robot: rl.robot.Robot | None = None

    def cancel(self) -> None:
        self._cancelled = True

    async def _check_cancelled(self) -> bool:
        if self._cancelled:
            logging.info("Acquisition cancelled")
            if self.robot is not None:
                await self.robot.stop()
            self._cancelled = False
            return True
        return False

    async def execute(
        self,
        cameras: list[rl.camera.Camera],
        trajectory: list[np.ndarray],
        bg_settings: BackgroundSettings | None = None,
        lights_settings: LightsSettings | None = None,
    ) -> AsyncGenerator[tuple[int, rl.camera.Camera], None]:
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

        assert cameras[0].robot is not None, "Cameras must be attached to a a robot"

        self.robot = cameras[0].robot
        robot = self.robot

        logging.debug(
            f"Executing trajectory with:\n\t- {cameras}\n\t- {bg_settings}\n\t- {lights_settings}"
        )

        if robot.home_pose is None:
            raise ValueError("Robot home pose is not set")

        await robot.move_to(robot.home_pose, timeout=50)

        logging.info("Reached home pose")

        await asyncio.sleep(0.2)
        if await self._check_cancelled():
            return

        for idx_trajectory, pose in enumerate(trajectory):
            # generate randomized bg and lights settings, to be re-used for all cameras

            bg_steps = self._scene.background.get_steps(bg_settings)
            light_steps = self._scene.lights.get_steps(lights_settings)

            for cam in cameras:
                if cam.extrinsic_matrix is not None:
                    robot_target = pose @ invert_homogeneous(cam.extrinsic_matrix)
                else:
                    logging.warning("Camera extrinsic matrix is not set, using pose from robot!")
                    robot_target = pose

                if await self._check_cancelled():
                    return

                await robot.move_to(robot_target, timeout=30)

                await asyncio.sleep(1.0)

                for bg_step, light_step in it.product(bg_steps, light_steps):
                    self._scene.background.set_step(bg_step)
                    self._scene.lights.set_step(light_step)

                    if await self._check_cancelled():
                        return

                    yield idx_trajectory, cam

        await robot.move_to(robot.home_pose, timeout=50)
