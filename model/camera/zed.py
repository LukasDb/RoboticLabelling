from .camera import Camera, CamFrame
import numpy as np
from typing import List
import cv2
import pyzed.sl as sl


class ZedCamera(Camera):
    RGB_H = 1080
    RGB_W = 1920
    DEPTH_H = 1280
    DEPTH_W = 720

    @staticmethod
    def get_available_devices() -> List["ZedCamera"]:
        cams = []
        print("Getting ZED devices...")
        print(
            "If you get segmentation fault here, reverse the USB type C cable on the ZED camera."
        )
        dev_list = sl.Camera.get_device_list()
        for dev in dev_list:  # list[DeviceProperties]
            print(f"Found device: {dev}")
            cams.append(ZedCamera(dev.serial_number))
        return cams

    def __init__(self, serial_number):
        self._serial_number = serial_number

        self.init_params = sl.InitParameters()
        self.init_params.sdk_verbose = 0  # 1 for verbose
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.camera_fps = 30
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
        self.init_params.coordinate_units = (
            sl.UNIT.METER
        )  # Use millimeter units (for depth measurements)
        self.init_params.set_from_serial_number(serial_number)

        self.device = sl.Camera()
        err = self.device.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Error opening ZED camera: ", err)
            exit()

        info = self.device.get_camera_information()
        super().__init__(str(info.camera_model) + f"-{self._serial_number}")

        self._rgb_buffer = sl.Mat()
        self._depth_buffer = sl.Mat()

    def get_frame(self) -> CamFrame:
        output = CamFrame()

        if self.device.grab() == sl.ERROR_CODE.SUCCESS:
            self.device.retrieve_image(self._rgb_buffer, sl.VIEW.LEFT)
            output.rgb = cv2.cvtColor(
                self._rgb_buffer.get_data(deep_copy=True), cv2.COLOR_BGR2RGB
            )
            self.device.retrieve_measure(self._depth_buffer, sl.MEASURE.DEPTH)
            output.depth = self._depth_buffer.get_data(deep_copy=True)

        return output

    @property
    def unique_id(self) -> str:
        return "zed_" + str(self._serial_number)

    def _set_hq_depth(self):
        self.is_hq_depth = True
        # TODO possibly change other parameters, too

    def _set_lq_depth(self):
        self.is_hq_depth = False
        # TODO possibly change other parameters, too
