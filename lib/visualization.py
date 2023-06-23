import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image


def create_image_widget(parent, img):
    h, w = img.shape[:2]
    prev = tk.Canvas(parent, width=w, height=h)
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img))
    prev.create_image(
        h // 2,
        w // 2,
        image=img_tk,
    )
    return prev, img_tk
