from utils import clamp

class Keyboard:

    def __init__(self) -> None:
        self.keyboard = {0: False}
        self.handlers = {}

    def key_down(self, key: str) -> None:
        self.keyboard[key.key] = True

    def key_up(self, key: str) -> None:
        self.keyboard[key.key] = False

    def get(self, key: Union[str, list, int]) -> Union[bool, str]:
        return self.keyboard.get(key, False)

    def get_axis(self, key: Union[str, T, typing.Hashable]):
        return self.handlers[key].value

    def add_handler(self, name: Union[str, typing.Callable, None], handler: Union[str, typing.Callable, None]) -> None:
        self.handlers[name] = handler

    def update(self, interval: Union[float, str, dict[str, int]]) -> None:
        for _, eachhandler in self.handlers.items():
            eachhandler.update(self, interval)

    def clear(self, axis: Union[int, str, dict[str, typing.Any]]) -> None:
        self.handlers.get(axis).value = 0

class ControlAxis:
    __pragma__('kwargs')

    def __init__(self, positive_key, negative_key, attack=1, decay=0, deadzone=0.02) -> None:
        self.positive = positive_key
        self.negative = negative_key
        self.attack = attack
        self.decay = decay
        self.deadzone = deadzone
        self.value = 0
    __pragma__('nokwargs')

    def update(self, keyboard, interval: Union[float, str, dict[str, int]]) -> None:
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