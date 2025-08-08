from utils import clamp
from typing import Dict

class Keyboard:
    def __init__(self):
        self.keyboard: Dict[int, bool] = {0: False}
        self.handlers: Dict[str, ControlAxis] = {}

    def key_down(self, key: int) -> None:
        self.keyboard[key] = True

    def key_up(self, key: int) -> None:
        self.keyboard[key] = False

    def get(self, key: int) -> bool:
        return self.keyboard.get(key, False)

    def get_axis(self, key: str) -> float:
        return self.handlers[key].value

    def add_handler(self, name: str, handler: 'ControlAxis') -> None:
        self.handlers[name] = handler

    def update(self, interval: float) -> None:
        for _, eachhandler in self.handlers.items():
            eachhandler.update(self, interval)

    def clear(self, axis: str) -> None:
        self.handlers[axis].value = 0

class ControlAxis:
    def __init__(self, positive_key: int, negative_key: int, attack: float = 1, decay: float = 0, deadzone: float = 0.02):
        self.positive = positive_key
        self.negative = negative_key
        self.attack = attack
        self.decay = decay
        self.deadzone = deadzone
        self.value = 0

    def update(self, keyboard: Keyboard, interval: float) -> None:
        self.value -= interval * self.decay * self.value
        dz = abs(self.value) < self.deadzone
        if keyboard.get(self.positive):
            dz = False
            self.value += interval * self.attack
        if keyboard.get(self.negative):
            dz = False
            self.value -= interval * self.attack
        if dz:
            self.value = 0
        else:
            self.value = clamp(self.value, -1, 1)
