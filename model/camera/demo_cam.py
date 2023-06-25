from .camera import Camera, CamFrame
import numpy as np
import cv2
from pathlib import Path
from itertools import cycle
import time


class DemoCam(Camera):
    def __init__(self, unique_id: str):
        super().__init__("MockCam")
        self._unique_id = unique_id
        self.mock_cam = "realsense_121622061798"
        # self.mock_cam = "realsense_f1120593"
        self.img_paths = cycle(Path(f"demo_data/images/{self.mock_cam}").glob("*.png"))

    def get_frame(self) -> CamFrame:
        img_path = next(self.img_paths)
        index = int(img_path.stem)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            pose_path = Path(f"demo_data/poses/{self.mock_cam}/{index}.txt")
            robot_pose = np.loadtxt(str(pose_path))
        except Exception:
            pose_path = Path(f"demo_data/poses/{self.mock_cam}/{index:04}.txt")
            robot_pose = np.loadtxt(str(pose_path))

        self.pose = robot_pose @ self.extrinsic_matrix
        time.sleep(0.5)
        return CamFrame(rgb=img)

    @property
    def unique_id(self) -> str:
        return self._unique_id
