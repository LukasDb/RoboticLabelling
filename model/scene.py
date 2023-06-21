from typing import Dict, List
from model.labelled_object import Labelledobject

class Scene:
    def __init__(self) -> None:
        self.objects: List[Labelledobject] = []