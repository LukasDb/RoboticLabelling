import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Tuple


class WidgetList(ttk.Frame):
    """Dynamic Table-like list of widgets"""

    def __init__(
        self,
        master,
        *,
        column_names: List[str] | None = None,
        columns: List[type[ttk.Widget | tk.Widget]],
        **kwargs
    ) -> None:
        super().__init__(
            master,
            **kwargs,
        )

        self.header_offset = 0
        if column_names is not None:
            self.header_offset = 2
            for i, c in enumerate(column_names):
                header = ttk.Label(self, text=c)
                header.grid(row=0, column=i, sticky=tk.EW)
            sep = ttk.Separator(self, orient=tk.HORIZONTAL)
            sep.grid(column=0, sticky=tk.EW, columnspan=len(column_names))

        self.rows: List[Tuple[ttk.Widget | tk.Widget]] = []
        self.widgets = columns

    def add_new_row(self, kwargs_list: None | List[dict] = None):
        if kwargs_list is None:
            kwargs_list = [{} for _ in self.widgets]

        row_tuple = []
        for i, (w, kwargs) in enumerate(zip(self.widgets, kwargs_list)):
            new_widget = w(self, **kwargs)
            new_widget.grid(sticky=tk.EW, row=len(self.rows) + self.header_offset, column=i)
            row_tuple.append(new_widget)

        self.rows.append(tuple(row_tuple))
        return row_tuple

    def pop(self, index):
        row = self.rows.pop(index)
        for w in row:
            w.destroy()

    def clear(self):
        for row in self.rows:
            for w in row:
                w.destroy()
        self.rows.clear()
