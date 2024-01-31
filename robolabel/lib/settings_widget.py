import tkinter as tk
from tkinter import ttk
import dataclasses
from typing import Any, ClassVar, Protocol


_SupportedVarTypes = tk.BooleanVar | tk.StringVar | tk.IntVar
_RowType = _SupportedVarTypes | list[_SupportedVarTypes] | tuple[_SupportedVarTypes]


class SettingsWidget(ttk.Frame):
    """Settings widget for arbitrary dataclasses"""

    def __init__(self, master: tk.Misc, dataclass: type, **kwargs: dict[str, Any]) -> None:
        defaults: dict[str, Any] = {"borderwidth": 2, "relief": tk.GROOVE}
        defaults.update(kwargs)
        super().__init__(master, **defaults)

        self.dataclass = dataclass
        assert hasattr(
            dataclass, "__dataclass_fields__"
        ), f"To create Settings for {dataclass}, it must be a dataclass"

        self.fields: dict[str, dataclasses.Field[Any]] = dataclass.__dataclass_fields__  # type: ignore
        self.vars: dict[str, _RowType] = {}

        self.fields = {k: v for k, v in self.fields.items() if not k.startswith("_")}
        self._private_vars: dict[str, _RowType] = {}

        pad = 10
        self.max_columns = 2
        title = ttk.Label(self, text=dataclass.__name__)
        for i, (name, field) in enumerate(self.fields.items()):
            label = ttk.Label(self, text=name)
            label.grid(row=i + 1, column=0, sticky=tk.EW, padx=pad, pady=5)
            self.vars[name] = self.create_entry_widget(self, i + 1, 1, field.type)

        for i in range(self.max_columns):
            self.columnconfigure(i, weight=1)

        # after we know max_columns
        title.grid(row=0, columnspan=self.max_columns, sticky=tk.EW)

        self.set_from_instance(dataclass())  # set defaults

    def create_entry_widget(self, master: tk.Misc, row: int, column: int, dtype: type) -> _RowType:
        """create a widget for a dataclass field"""
        var: _RowType
        widget: tk.Widget
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
        elif dtype is str:
            var = tk.StringVar()
            widget = ttk.Entry(master, textvariable=var)

        elif dtype.__origin__ is tuple:  # type: ignore
            widget = ttk.Frame(master)
            var = tuple(self.create_entry_widget(widget, row, column + offset, t) for offset, t in enumerate(dtype.__args__))  # type: ignore

        else:
            raise ValueError(f"Unsupported type {dtype}")

        widget.grid(row=row, column=column, sticky=tk.EW, padx=2)
        self.max_columns = max(self.max_columns, column + 1)
        return var

    def set_from_instance(self, dataclass_instance: Any) -> None:
        """set widget states from dataclass instance"""
        for k, v in dataclasses.asdict(dataclass_instance).items():
            if k.startswith("_"):
                self._private_vars[k] = v
            else:
                self._set_var(k, v)

    def _set_var(self, name: str, value: Any) -> None:
        var = self.vars[name]

        if isinstance(var, tuple) or isinstance(var, list):
            assert len(var) == len(value), f"Length mismatch for {name}"
            for _var, v in zip(var, value):
                _var.set(v)
        else:
            var.set(value)

    def get_instance(self) -> Any:
        """get dataclass instance from widget states"""
        return self.dataclass(
            **{k.name: self._get_widget(k) for k in self.fields.values()}, **self._private_vars
        )

    def _get_widget(self, field: dataclasses.Field[Any]) -> Any:
        var = self.vars[field.name]
        if isinstance(var, tuple) or isinstance(var, list):
            return tuple(dtype(v.get()) for v, dtype in zip(var, field.type.__args__))

        return field.type(var.get())
