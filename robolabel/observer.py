from enum import Enum, auto
from typing import Any, Set
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
    def __init__(self):
        self.__observers: Set["Observer"] = set()

    def register(self, observer: "Observer"):
        self.__observers.add(observer)

    def unregister(self, observer: "Observer"):
        self.__observers.remove(observer)

    def notify(self, event: Event, *args, **kwargs):
        logging.debug(f"[{self}]: {event}; {args}, {kwargs}")
        for observer in self.__observers:
            observer.update_observer(self, event, *args, **kwargs)


class Observer:
    def listen_to(self, subject: Observable):
        subject.register(self)

    def stop_listening(self, subject: Observable):
        subject.unregister(self)

    def update_observer(self, subject: Observable, event: Event, *args, **kwargs):
        pass
