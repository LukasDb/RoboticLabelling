import cv2
from .camera import Camera, CamFrame, DepthQuality
import numpy as np
import json
from pathlib import Path
import pyrealsense2 as rs
import logging
import threading

# mappings
occ_speed_map = {
    "very_fast": 0,
    "fast": 1,
    "medium": 2,
    "slow": 3,
    "wall": 4,
}
tare_accuracy_map = {
    "very_high": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}
scan_map = {
    "intrinsic": 0,
    "extrinsic": 1,
}
fl_adjust_map = {"right_only": 0, "both_sides": 1}


class Realsense(Camera):
    height = 1080
    width = 1920
    DEPTH_H = 720
    DEPTH_W = 1280

    @staticmethod
    def get_available_devices() -> list["Realsense"]:
        ctx = rs.context()  # type: ignore
        devices = ctx.query_devices()
        cams = []
        for dev in devices:
            logging.info(f"Found device: {dev.get_info(rs.camera_info.name)}")  # type: ignore
            serial_number = dev.get_info(rs.camera_info.serial_number)  # type: ignore
            try:
                cams.append(Realsense(serial_number))
            except Exception as e:
                logging.error(f"Could not initialize Realsense: {serial_number}: " + str(e))

        return cams

    def __init__(self, serial_number: str) -> None:
        self._serial_number = serial_number
        self._depth_quality = DepthQuality.INFERENCE
        self._lock = threading.Lock()

        self._pipeline = rs.pipeline()  # type: ignore
        self._config = config = rs.config()  # type: ignore
        config.enable_device(self._serial_number)

        pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)  # type: ignore
        pipeline_profile = config.resolve(pipeline_wrapper)

        self.device = pipeline_profile.get_device()
        super().__init__(self.device.get_info(rs.camera_info.name))  # type: ignore

        if self.device.get_info(rs.camera_info.firmware_version) != self.device.get_info(  # type: ignore
            rs.camera_info.recommended_firmware_version  # type: ignore
        ):
            logging.warn(f"Camera {self.name} firmware is out of date")

        config.enable_stream(rs.stream.depth, self.DEPTH_W, self.DEPTH_H, rs.format.z16, 30)  # type: ignore
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, 30)  # type: ignore
        self.align_to_rgb = rs.align(rs.stream.color)  # type: ignore

        self.temporal_filter = rs.temporal_filter(  # type: ignore
            smooth_alpha=0.4,
            smooth_delta=20,
            persistence_control=3,
        )

        self.spatial_filter = rs.spatial_filter(  # type: ignore
            smooth_alpha=0.5,
            smooth_delta=20,
            magnitude=1,
            hole_fill=2,
        )

        if "D415" in self.name:
            profile_path = Path("realsense_profiles/d415_HQ.json")
        elif "D435" in self.name:
            profile_path = Path("realsense_profiles/d435_HQ.json")
        else:
            raise Exception(f"Unsupported camera: {self.name}")

        logging.debug(f"[{self.name}] Loading configuration: {profile_path.name}")
        rs.rs400_advanced_mode(self.device).load_json(profile_path.read_text())  # type: ignore

        self.sensor = self.device.first_depth_sensor()
        self.depth_scale = self.sensor.get_depth_scale()

        self.sensor.set_option(rs.option.hdr_enabled, False)  # type: ignore

        # Start streaming
        self.is_started = False
        self._pipeline.start(self._config)
        self.is_started = True

    def get_frame(self, depth_quality: DepthQuality) -> CamFrame:
        with self._lock:
            if depth_quality != DepthQuality.UNCHANGED and depth_quality != self._depth_quality:
                self._depth_quality = depth_quality
                # if depth_quality is DepthQuality.GT:
                #     self.sensor.set_option(rs.option.hdr_enabled, True)  # type: ignore
                # elif depth_quality is DepthQuality.INFERENCE:
                #     self.sensor.set_option(rs.option.hdr_enabled, False)  # type: ignore

            output = CamFrame()
            if not self.is_started:
                return output
            try:
                frames = self._pipeline.wait_for_frames()
            except Exception as e:
                logging.error(f"Could not get camera frame: {e}")
                self.device.hardware_reset()
                self._pipeline.start(self._config)
                return output

            # makes depth frame same resolution as rgb frame
            aligned_frames = self.align_to_rgb.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                logging.warn("Could not get camera frame")
                return output

            if False:  # self._depth_quality in [DepthQuality.INFERENCE, DepthQuality.GT]:
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.spatial_filter.process(depth_frame)

            depth_image = np.asarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            color_image = np.asarray(color_frame.get_data())  # type: ignore

            output.depth = depth_image
            output.rgb = color_image

            return output

    @property
    def unique_id(self) -> str:
        return "realsense_" + str(self._serial_number)

    def run_self_calibration(self) -> None:
        """run on-chip calibration of realsense."""
        logging.warn(
            "Full automatic calibration is not supported yet. Please manually calibrate the camera (from realsense-viewer)."
        )
        logging.warn("If using tare calibration, check the distance to the monitor")
        return

        self._pipeline.stop()
        self.is_started = False

        self.device.hardware_reset()

        speed = "slow"
        data = {
            "calib type": 0,
            "speed": occ_speed_map[speed],
            "scan parameter": scan_map["intrinsic"],
            "white_wall_mode": 1 if speed == "wall" else 0,
        }

        cfg = rs.config()  # type: ignore
        cfg.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)  # type: ignore
        cfg.enable_device(self._serial_number)

        pipe = rs.pipeline()  # type: ignore

        pp = pipe.start(cfg)
        dev = pp.get_device()

        args = json.dumps(data)

        def progress_callback(progress):
            print(f"\rProgress  {progress}% ... ", end="\r")

        try:
            print("Starting On-Chip calibration...")
            print(f"\tSpeed:\t{speed}")
            auto_device = dev.as_auto_calibrated_device()
            table, health = auto_device.run_on_chip_calibration(args, progress_callback, 30000)
            print("On-Chip calibration finished")
            print(f"\tHealth: {health}")
            auto_device.set_calibration_table(table)
            auto_device.write_calibration()
        finally:
            pipe.stop()

        self._pipeline.start(self._config)
        self.is_started = True
