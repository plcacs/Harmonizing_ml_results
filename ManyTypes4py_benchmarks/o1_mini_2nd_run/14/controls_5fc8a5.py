from typing import Dict, Any
from utils import clamp

class Keyboard:

    def __init__(self) -> None:
        self.keyboard: Dict[Any, bool] = {0: False}
        self.handlers: Dict[str, 'ControlAxis'] = {}

    def key_down(self, key: Any) -> None:
        self.keyboard[key.key] = True

    def key_up(self, key: Any) -> None:
        self.keyboard[key.key] = False

    def get(self, key: Any) -> bool:
        return self.keyboard.get(key, False)

    def get_axis(self, key: str) -> float:
        return self.handlers[key].value

    def add_handler(self, name: str, handler: 'ControlAxis') -> None:
        self.handlers[name] = handler

    def update(self, interval: float) -> None:
        for _, eachhandler in self.handlers.items():
            eachhandler.update(self, interval)

    def clear(self, axis: str) -> None:
        handler = self.handlers.get(axis)
        if handler:
            handler.value = 0.0

class ControlAxis:
    __pragma__('kwargs')

    def __init__(
        self, 
        positive_key: Any, 
        negative_key: Any, 
        attack: float = 1.0, 
        decay: float = 0.0, 
        deadzone: float = 0.02
    ) -> None:
        self.positive: Any = positive_key
        self.negative: Any = negative_key
        self.attack: float = attack
        self.decay: float = decay
        self.deadzone: float = deadzone
        self.value: float = 0.0
    __pragma__('nokwargs')

    def update(self, keyboard: Keyboard, interval: float) -> None:
        self.value -= interval * self.decay * self.value
        dz: bool = abs(self.value) < self.deadzone
        if keyboard.get(self.positive):
            dz = False
            self.value += interval * self.attack
        if keyboard.get(self.negative):
            dz = False
            self.value -= interval * self.attack
        if dz:
            self.value = 0.0
        else:
            self.value = clamp(self.value, -1.0, 1.0)
