from .camera import Camera, CamFrame
import numpy as np
from typing import List
import cv2
import pyrealsense2 as rs
import logging


class Realsense(Camera):
    height = 1080
    width = 1920
    DEPTH_H = 720
    DEPTH_W = 1280

    @staticmethod
    def get_available_devices() -> List["Realsense"]:
        ctx = rs.context()  # type: ignore
        devices = ctx.query_devices()
        cams = []
        for dev in devices:
            logging.info(f"Found device: {dev.get_info(rs.camera_info.name)}")  # type: ignore
            serial_number = dev.get_info(rs.camera_info.serial_number)  # type: ignore
            cams.append(Realsense(serial_number))
        return cams

    def __init__(self, serial_number):
        self.is_hq_depth = True
        self._serial_number = serial_number

        self._pipeline = rs.pipeline()  # type: ignore
        self._config = config = rs.config()  # type: ignore
        # config.enable_device(self._serial_number)

        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)  # type: ignore
        try:
            pipeline_profile = config.resolve(pipeline_wrapper)
        except RuntimeError:
            logging.error(
                "Realsense Device is not connected. Please connect the device and try again."
            )
            exit()
        self.device = pipeline_profile.get_device()
        super().__init__(self.device.get_info(rs.camera_info.name))  # type: ignore

        config.enable_stream(rs.stream.depth, self.DEPTH_W, self.DEPTH_H, rs.format.z16, 30)  # type: ignore
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)  # type: ignore
        self.align_to_rgb = rs.align(rs.stream.color)  # type: ignore

        self.temporal_filter = rs.temporal_filter(  # type: ignore
            smooth_alpha=0.2,
            smooth_delta=5,
            persistence_control=2,
        )

        self.spatial_filter = rs.spatial_filter(  # type: ignore
            smooth_alpha=0.5,
            smooth_delta=20,
            magnitude=2,
            hole_fill=1,
        )

        self.depth_scale = self.device.first_depth_sensor().get_depth_scale()

        # Start streaming
        self.is_started = False
        try:
            self._pipeline.start(self._config)
            self.is_started = True
        except RuntimeError as e:
            logging.error(f"Could not start camera stream ({self.name})")
            logging.error(e)

    def get_frame(self) -> CamFrame:
        output = CamFrame()
        if not self.is_started:
            return output
        frames = self._pipeline.wait_for_frames()
        # makes depth frame same resolution as rgb frame
        frames = self.align_to_rgb.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            logging.warn("Could not get camera frame")
            return output

        if self.is_hq_depth:
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)

        depth_image = np.asarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        color_image = cv2.cvtColor(np.asarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)  # type: ignore

        output.depth = depth_image
        output.rgb = color_image

        return output

    @property
    def unique_id(self) -> str:
        return "realsense_" + str(self._serial_number)

    def _set_hq_depth(self):
        self.is_hq_depth = True
        # TODO possibly change other parameters, too

    def _set_lq_depth(self):
        self.is_hq_depth = False
        # TODO possibly change other parameters, too
