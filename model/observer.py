from enum import Enum, auto
from typing import List


class Event(Enum):
    CAMERA_CALIBRATED = auto()


class Subject:
    def __init__(self):
        self.__observers: List["Observer"] = []

    def register(self, observer):
        self.__observers.append(observer)

    def notify(self, event: Event, *args, **kwargs):
        for observer in self.__observers:
            observer.update(self, event, *args, **kwargs)


class Observer:
    def __init__(self, subject: Subject = None):
        if subject is not None:
            subject.register(self)

    def listen_to(self, subject: Subject):
        subject.register(self)

    def update(self, subject: Subject, event: Event, *args, **kwargs):
        pass
