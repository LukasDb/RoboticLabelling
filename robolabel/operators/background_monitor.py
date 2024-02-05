from typing import Any
import numpy as np
import numpy.typing as npt
import tkinter as tk
from PIL import Image
import screeninfo
import pathlib
import cv2
import logging
from dataclasses import dataclass

from robolabel.observer import Event, Observable, Observer
import robolabel as rl


class DemoMonitor:
    height = 2160
    width = 3840
    diagonal_16_by_9 = float(np.linalg.norm((16, 9)))
    width_mm = 16 / diagonal_16_by_9 * 800  # 0.69726
    height_mm = 9 / diagonal_16_by_9 * 800  # 0.3922


@dataclass
class BackgroundSettings:
    """Settings for the background monitor randomization"""

    use_backgrounds: bool = True
    n_steps: int = 5
    backgrounds_path: str = "./backgrounds"


class BackgroundMonitor(rl.Entity, Observer):
    """Background Monitor, the pose refers to the center of the monitor
    with coordinate system as if it was a cv2 image
    """

    def __init__(self) -> None:
        rl.Entity.__init__(self, "Background Monitor")
        Observer.__init__(self)
        # get secondary monitor info
        self.window = tk.Toplevel()
        self.window.title("Background Monitor -> MOVE THIS WINDOW TO SECONDARY MONITOR")

        self._is_demo_mode = False
        # create fullscreen canvas
        self.canvas = rl.lib.ResizableImage(self.window, mode="zoom")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # move window to second screen
        self.window.geometry("640x480")
        self.window.geometry("+0+0")

        self._monitor = self._get_current_monitor()
        self.setup_window(set_fullscreen=False)

    def update_observer(
        self, subject: Observable, event: Event, *args: Any, **kwargs: Any
    ) -> None:
        if event == Event.CAMERA_SELECTED:
            self._is_demo_mode = isinstance(kwargs["camera"], rl.camera.DemoCam)
        return super().update_observer(subject, event, *args, **kwargs)

    def _get_current_monitor(self) -> screeninfo.Monitor | DemoMonitor:
        if self._is_demo_mode:
            logging.warn("USING MOCK MONITOR DIMENSIONS")
            return DemoMonitor()

        monitors = screeninfo.get_monitors()
        x, y = self.window.winfo_x(), self.window.winfo_y()
        monitor = None
        for m in reversed(monitors):
            if m.x <= x < m.width + m.x and m.y <= y < m.height + m.y:
                monitor = m
                break
        logging.debug(f"Using monitor {monitor}")
        assert monitor is not None, "Could not find monitor"
        return monitor

    @property
    def screen_width(self) -> int:
        return self._monitor.width

    @property
    def screen_height(self) -> int:
        return self._monitor.height

    @property
    def screen_width_m(self) -> float:
        width_mm = self._monitor.width_mm
        assert width_mm is not None, "Could not get monitor width in mm"
        return width_mm / 1000.0

    @property
    def screen_height_m(self) -> float:
        height_mm = self._monitor.height_mm
        assert height_mm is not None, "Could not get monitor height in mm"
        return height_mm / 1000.0

    def get_steps(
        self, settings: BackgroundSettings | None
    ) -> list[dict[str, pathlib.Path] | None]:
        if settings is None or not settings.use_backgrounds:
            return [None]

        bg_paths = np.random.choice(
            list(str(x) for x in pathlib.Path(settings.backgrounds_path).iterdir()),
            size=settings.n_steps,
        )

        return [{"background": p} for p in bg_paths]

    def set_step(self, step: dict[str, pathlib.Path] | None) -> None:
        if step is None:
            return
        if "background" in step:
            bg_path = pathlib.Path(step["background"])
            logging.debug(f"Setting background monitor to {bg_path.name}")
            self._load_image_to_full_canvas(bg_path)

    def set_textured(self) -> None:
        # from https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
        textured_folder = pathlib.Path("./intel_textured_patterns")
        textured_paths = list(textured_folder.iterdir())
        # choose the one that is closest to the current resolution
        textured_widths = [int(p.name.split("_")[5]) - self.width for p in textured_paths]
        selected = np.argmin(np.abs(textured_widths))
        textured_path = textured_paths[selected]
        self._load_image_to_full_canvas(textured_path)

    def _load_image_to_full_canvas(self, path: pathlib.Path) -> None:
        self.setup_window(set_fullscreen=True)
        with path.open("rb") as f:
            bg = np.asarray(Image.open(f))
        # scale image to fill the screen
        bg = cv2.resize(  # type: ignore
            bg,
            (
                int(self.window.winfo_width()),
                int(self.window.winfo_height()),
            ),
        )
        self.set_image(bg)

    def setup_window(self, set_fullscreen: bool = True) -> None:
        # get window size
        # self.window.geometry("+0+0")
        if set_fullscreen:
            self.window.attributes("-fullscreen", True)
        self.window.update()
        self.width = self.window.winfo_width()
        self.height = self.window.winfo_height()

        self._monitor = self._get_current_monitor()

        logging.debug(
            f"Setting window up for screen with ({self.screen_width}, {self.screen_height}) pixels"
        )

        logging.debug(
            f"Setting window up for screen with ({self.screen_width_m}, {self.screen_height_m}) meters"
        )

    def set_image(self, image: npt.NDArray[np.float64]) -> None:
        """Set the image of the background monitor"""
        # self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(image))
        # self.canvas.itemconfig(self.image_container, image=self.image_tk)
        self.canvas.set_image(image)
        self.canvas.update()

    def visualize_monitor_in_camera_view(
        self,
        rgb: np.ndarray,
        intrinsic: np.ndarray,
        dist_coeffs: np.ndarray,
        cam2monitor: np.ndarray,
        color: tuple[int, int, int] | np.ndarray = (0, 255, 0),
    ) -> None:
        """Draw the background monitor on the rgb image"""
        rvec, _ = cv2.Rodrigues(cam2monitor[:3, :3])  # type: ignore
        tvec = cam2monitor[:3, 3]
        cv2.drawFrameAxes(rgb, intrinsic, dist_coeffs, rvec, tvec, 0.1, 3)  # type: ignore

        if self.screen_height_m is None or self.screen_width_m is None:
            return

        half_w = self.screen_width_m / 2.0
        half_h = self.screen_height_m / 2.0

        screen_corners = np.array(
            [
                [-half_w, -half_h, 0, 1],
                [half_w, -half_h, 0, 1],
                [half_w, half_h, 0, 1],
                [-half_w, half_h, 0, 1],
            ]
        )

        screen_corners_cam = cam2monitor @ screen_corners.T
        img_points, _ = cv2.projectPoints(  # type: ignore
            screen_corners_cam.T[:, :3],
            np.zeros(
                3,
            ),
            np.zeros(
                3,
            ),
            intrinsic,
            dist_coeffs,
        )

        img_points = img_points.astype(np.int32)

        cv2.drawContours(  # type: ignore
            rgb,
            [img_points],
            -1,  # draw all contours
            color,
            3,  # negative == filled
            lineType=cv2.LINE_AA,  # type: ignore
        )
