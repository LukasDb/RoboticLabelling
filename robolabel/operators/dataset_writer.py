from ..labelled_object import LabelledObject


class DatasetWriter:
    def __init__(self) -> None:
        self._active_objects: list[LabelledObject] = []

    def set_objects(self, objects: list[LabelledObject]) -> None:
        self._active_objects = objects
