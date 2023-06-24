from .camera import Camera, CamFrame
import numpy as np
import cv2
import time


class RealsenseD415(Camera):
    def __init__(self):
        super().__init__("RealsenseD415")
        self.frame_count = 0

    def get_frame(self) -> CamFrame:
        rgb = np.random.uniform(size=(1080, 1920, 3), high=255).astype(np.uint8)
        # write frame number on rgb
        cv2.putText(
            rgb,
            f"Frame {self.frame_count}",
            (200, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        self.frame_count += 1
        # time.sleep(1 / 60.0)
        return CamFrame(rgb=rgb)

    @property
    def unique_id(self) -> str:
        return "1234"
