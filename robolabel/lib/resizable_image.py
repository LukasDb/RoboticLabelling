import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import numpy.typing as npt
import cv2
from typing import Literal, Any


class ResizableImage(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        image: None | npt.NDArray[np.uint8] = None,
        mode: Literal["stretch", "zoom", "fit"] = "fit",
        **kwargs: Any,
    ) -> None:
        tk.Canvas.__init__(self, master, kwargs)
        self._canvas_img = self.create_image(0, 0, anchor=tk.NW)
        self._img: npt.NDArray[np.uint8] | None = None
        self.mode = mode
        if image is not None:
            self._img = image.copy()
        self.bind(sequence="<Configure>", func=self._on_refresh)

    def _on_refresh(self, event: None | tk.Event = None) -> None:
        if event is not None:
            self.widget_width = int(event.width)
            self.widget_height = int(event.height)

        if self._img is None or not hasattr(self, "widget_width"):
            return

        img_width = self._img.shape[1]
        img_height = self._img.shape[0]
        if self.mode == "fit":
            scale = min(self.widget_width / img_width, self.widget_height / img_height)
            scaled_img_width = int(img_width * scale)
            scaled_img_height = int(img_height * scale)
        elif self.mode == "zoom":
            scale = max(self.widget_width / img_width, self.widget_height / img_height)
            scaled_img_width = int(img_width * scale)
            scaled_img_height = int(img_height * scale)

        elif self.mode == "stretch":
            scaled_img_width = self.widget_width
            scaled_img_height = self.widget_height
        else:
            raise ValueError(f"Unknown image mode: {self.mode}")

        img_resized = cv2.resize(  # type: ignore
            self._img.copy(), (scaled_img_width, scaled_img_height)
        )

        self._img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.itemconfig(self._canvas_img, image=self._img_tk)  # , anchor=tk.CENTER)

    def set_image(self, image: np.ndarray) -> None:
        self._img = image
        if self._canvas_img not in self.children:
            self._canvas_img = self.create_image(0, 0, anchor=tk.NW)
        self._on_refresh()  # resize to canvas

    def clear_image(self) -> None:
        if self.winfo_exists():
            self.delete(self._canvas_img)
