from model.scene import Scene

class State:
    # hold current application state
    def __init__(self) -> None:
        self.current_scene: Scene = None