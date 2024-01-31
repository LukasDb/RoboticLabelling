from enum import Enum, auto
from typing import Any
import logging


class Event(Enum):
    CAMERA_ADDED = auto()
    CAMERA_CALIBRATED = auto()
    CAMERA_ATTACHED = auto()
    CAMERA_SELECTED = auto()

    OBJECT_ADDED = auto()
    OBJECT_REGISTERED = auto()
    OBJECT_SELECTED = auto()
    OBJECT_REMOVED = auto()
    ROBOT_ADDED = auto()
    MODE_CHANGED = auto()


class Observable:
    def __init__(self) -> None:
        self.__observers: set["Observer"] = set()

    def register(self, observer: "Observer") -> None:
        self.__observers.add(observer)

    def unregister(self, observer: "Observer") -> None:
        self.__observers.remove(observer)

    def notify(self, event: Event, *args: Any, **kwargs: Any) -> None:
        logging.debug(f"[{self}]: {event}; {args}, {kwargs}")
        for observer in self.__observers:
            observer.update_observer(self, event, *args, **kwargs)


class Observer:
    def listen_to(self, subject: Observable) -> None:
        subject.register(self)

    def stop_listening(self, subject: Observable) -> None:
        subject.unregister(self)

    def update_observer(
        self, subject: Observable, event: Event, *args: Any, **kwargs: Any
    ) -> None:
        pass
