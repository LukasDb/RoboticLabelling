import tkinter as tk
from tkinter import ttk
from model.scene import Scene


class ViewAcquisition(tk.Frame):
    def __init__(self, parent, scene: Scene) -> None:
        tk.Frame.__init__(self, parent)
        self.scene = scene

        self.title = ttk.Label(self, text="3. Acquisition")
        self.title.grid()
