import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import cv2


class ResizableImage(tk.Canvas):
    def __init__(self, master, image=None, **kwargs):
        tk.Canvas.__init__(self, master, **kwargs)
        self._canvas_img = self.create_image(0, 0, anchor=tk.NW)
        self._img_tk: ImageTk.PhotoImage = None
        self._img: np.ndarray = None
        if image is not None:
            self._img = image.copy()
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event=None):
        if event is not None:
            self.widget_width = event.width
            self.widget_height = event.height

        if self._img is None or not hasattr(self, "widget_width"):
            return

        img_width = self._img.shape[1]
        img_height = self._img.shape[0]

        scale = min(self.widget_width / img_width, self.widget_height / img_height)
        scaled_img_width = int(img_width * scale)
        scaled_img_height = int(img_height * scale)
        img_resized = cv2.resize(
            self._img.copy(), (scaled_img_width, scaled_img_height)
        )

        self._img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized))
        self.itemconfig(self._canvas_img, image=self._img_tk)  # , anchor=tk.CENTER)

    def set_image(self, image):
        self._img = image.copy()
        if self._canvas_img not in self.children:
            self._canvas_img = self.create_image(0, 0, anchor=tk.NW)
        self._on_resize()  # resize to canvas

    def clear_image(self):
        if self.winfo_exists():
            self.delete(self._canvas_img)
