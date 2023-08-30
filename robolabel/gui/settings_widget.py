import tkinter as tk
from tkinter import ttk
import dataclasses


class SettingsWidget(ttk.Frame):
    """Settings widget for arbitrary dataclasses"""

    def __init__(self, master, dataclass, **kwargs) -> None:
        super().__init__(master, **kwargs)

        self.dataclass = dataclass
        self.fields: dict[str, dataclasses.Field] = dataclass.__dataclass_fields__
        self.vars: list[tk.Variable] = {}

        if not hasattr(dataclass, "__dataclass_fields__"):
            raise ValueError("dataclass must be a dataclass")

        for i, (name, field) in enumerate(self.fields.items()):
            label = ttk.Label(self, text=name)
            label.grid(row=i, column=0, sticky=tk.EW)
            self.vars[name] = self.create_entry_widget(self, i, 1, field.type)

    def create_entry_widget(self, master, row: int, column: int, dtype: type):
        if dtype is int:
            var = tk.StringVar()
            widget = ttk.Spinbox(master, from_=-1000, to=1000, textvariable=var)
        elif dtype is float:
            var = tk.StringVar()
            widget = ttk.Spinbox(
                master, from_=-1000.0, to=1000.0, increment=0.01, textvariable=var
            )
        elif dtype is bool:
            var = tk.BooleanVar()
            widget = tk.Checkbutton(master, variable=var)
        elif dtype.__origin__ is tuple:
            widget = ttk.Frame(master)
            var = []
            for offset, t in enumerate(dtype.__args__):
                var.append(self.create_entry_widget(widget, row, column + offset, t))

        else:
            print(f"could not create widget for field {dtype}")
            return
        widget.grid(row=row, column=column, sticky=tk.EW)
        return var

    def set_from_instance(self, dataclass_instance):
        """set widget states from dataclass instance"""
        for k, v in dataclasses.asdict(dataclass_instance).items():
            self._set_widget(k, v)

    def _set_widget(self, name, value):
        var = self.vars[name]
        if isinstance(var, tuple) or isinstance(var, list):
            for _var, v in zip(var, value):
                _var.set(v)
        else:
            var.set(value)

    def get_instance(self):
        """get dataclass instance from widget states"""
        return self.dataclass(**{k.name: self._get_widget(k) for k in self.fields.values()})

    def _get_widget(self, field: dataclasses.Field):
        var = self.vars[field.name]
        if isinstance(var, tuple) or isinstance(var, list):
            return tuple(dtype(v.get()) for v, dtype in zip(var, field.type.__args__))
        return field.type(var.get())
