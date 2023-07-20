from enum import Enum, auto
from typing import List


class Event(Enum):
    CAMERA_ADDED = auto()
    CAMERA_CALIBRATED = auto()
    CAMERA_ATTACHED = auto()
    OBJECT_ADDED = auto()
    ROBOT_ADDED = auto()


class Subject:
    def __init__(self):
        self.__observers: List["Observer"] = []

    def register(self, observer):
        self.__observers.append(observer)

    def notify(self, event: Event, *args, **kwargs):
        for observer in self.__observers:
            observer.update_observer(self, event, *args, **kwargs)


class Observer:
    def __init__(self, subject: Subject = None):
        if subject is not None:
            subject.register(self)

    def listen_to(self, subject: Subject):
        subject.register(self)

    def update_observer(self, subject: Subject, event: Event, *args, **kwargs):
        pass
