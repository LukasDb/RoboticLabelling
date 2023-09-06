import numpy as np
import tkinter as tk
from .entity import Entity
from PIL import Image, ImageTk
import screeninfo
import pathlib
import cv2
import logging
from robolabel.lib.geometry import *
from robolabel.lib.resizable_image import ResizableImage
from dataclasses import dataclass


@dataclass
class BackgroundSettings:
    """Settings for the background monitor randomization"""

    use_backgrounds: bool = True
    n_steps: int = 5
    backgrounds_path: str = "./backgrounds"


class BackgroundMonitor(Entity):
    """Background Monitor, the pose refers to the center of the monitor
    with coordinate system as if it was a cv2 image
    """

    def __init__(self):
        super().__init__("Background Monitor")
        # get secondary monitor info
        self.window = tk.Toplevel()
        self.window.title("Background Monitor -> MOVE THIS WINDOW TO SECONDARY MONITOR")

        # create fullscreen canvas
        self.canvas = ResizableImage(self.window, mode="zoom")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # move window to second screen
        self.window.geometry("640x480")
        self.window.geometry("+0+0")
        self.setup_window(set_fullscreen=False)

    def get_steps(self, settings: BackgroundSettings | None) -> list[dict | None]:
        if settings is None:
            settings = BackgroundSettings()
        if not settings.use_backgrounds:
            return [None]

        bg_paths = np.random.choice(
            list(str(x) for x in pathlib.Path(settings.backgrounds_path).iterdir()),
            size=settings.n_steps,
        )

        return [{"background": p} for p in bg_paths]

    def set_step(self, step: dict | None) -> None:
        if step is None:
            return
        if "background" in step:
            bg_path = pathlib.Path(step["background"])
            logging.debug(f"Setting background monitor to {bg_path.name}")
            self._load_image_to_full_canvas(bg_path)

    def set_textured(self):
        # from https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
        textured_folder = pathlib.Path("./intel_textured_patterns")
        textured_paths = list(textured_folder.iterdir())
        # choose the one that is closest to the current resolution
        textured_widths = [int(p.name.split("_")[5]) - self.width for p in textured_paths]
        selected = np.argmin(np.abs(textured_widths))
        textured_path = textured_paths[selected]
        self._load_image_to_full_canvas(textured_path)

    def _load_image_to_full_canvas(self, path: pathlib.Path) -> None:
        self.window.attributes("-fullscreen", True)
        self.window.update()
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

    def setup_window(self, set_fullscreen=True):
        # get window size
        # self.window.geometry("+0+0")
        if set_fullscreen:
            self.window.attributes("-fullscreen", True)
        self.window.update()
        self.width = self.window.winfo_width()
        self.height = self.window.winfo_height()

        monitors = screeninfo.get_monitors()

        x, y = self.window.winfo_x(), self.window.winfo_y()
        monitor = None
        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                monitor = m
                break

        if monitor is None:
            logging.error("Could not find monitor for background monitor")
            return

        # get screen size
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        logging.debug(
            f"Setting window up for screen with ({self.screen_width}, {self.screen_height}) pixels"
        )

        # get screen size in mm
        assert monitor.width_mm is not None, "Could not measure monitor!"
        assert monitor.height_mm is not None, "Could not measure monitor!"
        self.screen_width_m = monitor.width_mm / 1000.0
        self.screen_height_m = monitor.height_mm / 1000.0
        logging.debug(
            f"Setting window up for screen with ({self.screen_width_m}, {self.screen_height_m}) meters"
        )

    def set_image(self, image: np.ndarray) -> None:
        """Set the image of the background monitor"""
        # self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(image))
        # self.canvas.itemconfig(self.image_container, image=self.image_tk)
        self.canvas.set_image(image)

    def draw_on_rgb(self, rgb, intrinsic, dist_coeffs, cam2monitor, color=(0, 255, 0)):
        """Draw the background monitor on the rgb image"""
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
