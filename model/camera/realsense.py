from .camera import Camera, CamFrame
import numpy as np
from typing import List
import cv2
import pyrealsense2 as rs


class Realsense(Camera):
    RGB_H = 1080
    RGB_W = 1920
    DEPTH_H = 1280
    DEPTH_W = 720

    @staticmethod
    def get_available_devices() -> List["Realsense"]:
        ctx = rs.context()
        devices = ctx.query_devices()
        cams = []
        for dev in devices:
            print("Found device: ", dev.get_info(rs.camera_info.name))
            serial_number = dev.get_info(rs.camera_info.serial_number)
            cams.append(Realsense(serial_number))
        return cams

    def __init__(self, serial_number):
        super().__init__("Realsense D415")

        self.is_hq_depth = True
        self._serial_number = serial_number

        self._pipeline = rs.pipeline()
        self._config = config = rs.config()
        config.enable_device(self._serial_number)

        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        try:
            pipeline_profile = config.resolve(pipeline_wrapper)
        except RuntimeError:
            print(
                "Realsense Device is not connected. Please connect the device and try again."
            )
            exit()
        self.device = pipeline_profile.get_device()

        config.enable_stream(
            rs.stream.depth, self.DEPTH_W, self.DEPTH_H, rs.format.z16, 30
        )
        config.enable_stream(
            rs.stream.color, self.RGB_W, self.RGB_H, rs.format.bgr8, 30
        )
        self.align_to_rgb = rs.align(rs.stream.color)

        self.temporal_filter = rs.temporal_filter(
            smooth_alpha=0.2,
            smooth_delta=5,
            persistence_control=2,
        )

        self.spatial_filter = rs.spatial_filter(
            smooth_alpha=0.5,
            smooth_delta=20,
            magnitude=2,
            hole_fill=1,
        )

        self.depth_scale = self.device.first_depth_sensor().get_depth_scale()

        # Start streaming
        self._pipeline.start(self._config)

    def get_frame(self) -> CamFrame:
        output = CamFrame()

        frames = self._pipeline.wait_for_frames()
        # makes depth frame same resolution as rgb frame
        frames = self.align_to_rgb.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            print("Could not get camera frame")
            return output

        if self.is_hq_depth:
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)

        depth_image = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        )
        color_image = cv2.cvtColor(
            np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB
        )

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
