from dataclasses import dataclass
import logging


@dataclass
class LightsSettings:
    """settings for ambient lighting"""

    use_lights: bool = False
    n_steps: int = 3


class LightsController:
    """control ambient lighting"""

    def get_steps(self, settings: LightsSettings | None) -> list[dict]:
        if settings is None:
            settings = LightsSettings()
        if not settings.use_lights:
            return [
                {},
            ]

        return [{"light1": 0, "light2": 0, "light3": 0}] * 3

    def set_step(self, step: dict | None) -> None:
        if step is None:
            return
        pass
