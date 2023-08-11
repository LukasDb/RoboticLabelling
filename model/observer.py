from enum import Enum, auto
from typing import List
import logging


class Event(Enum):
    CAMERA_ADDED = auto()
    CAMERA_CALIBRATED = auto()
    CAMERA_ATTACHED = auto()
    OBJECT_ADDED = auto()
    OBJECT_REGISTERED = auto()
    ROBOT_ADDED = auto()


class Observable:
    def __init__(self):
        self.__observers: List["Observer"] = []

    def register(self, observer):
        self.__observers.append(observer)

    def notify(self, event: Event, *args, **kwargs):
        logging.debug(f"[{self}]: {event}")
        for observer in self.__observers:
            observer.update_observer(self, event, *args, **kwargs)


class Observer:
    def listen_to(self, subject: Observable):
        subject.register(self)

    def update_observer(self, subject: Observable, event: Event, *args, **kwargs):
        pass
