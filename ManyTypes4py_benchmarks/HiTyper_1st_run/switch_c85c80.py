from typing import List

class SwitchState:
    ON = True
    OFF = False

    @staticmethod
    def to_state(string: str):
        return SwitchState.ON if string == 'ON' else SwitchState.OFF

    @staticmethod
    def to_string(state: Union[dict[str, str], str, dict]) -> typing.Text:
        return 'ON' if state else 'OFF'

class Switch:
    switches = []

    def __init__(self, name: Union[str, None], title: Union[None, str]=None, description: Union[None, str, list[str]]=None, state: Any=SwitchState.ON) -> None:
        self.group = 'switch'
        self.name = name
        self.title = title
        self.description = description
        self.state = state
        Switch.switches.append(self)

    def serialize(self) -> dict[typing.Text, typing.Text]:
        return {'group': 'switch', 'name': self.name, 'type': 'text', 'title': self.title, 'description': self.description, 'value': 'ON' if self.is_on else 'OFF'}

    def __repr__(self) -> str:
        return 'Switch(name={!r}, description={!r}, state={!r})'.format(self.name, self.description, SwitchState.to_string(self.state))

    @classmethod
    def find_by_name(cls: Union[str, None, typing.Type], name: Union[str, tuple[typing.Union[str,...]]]) -> None:
        for s in Switch.switches:
            if s.name == name:
                return s
        return

    @classmethod
    def find_all(cls: Union[str, typing.Callable[typing.Any, T], bool]):
        return Switch.switches

    def set_state(self, state: Union[dict, str, dict[str, typing.Any]]) -> None:
        self.state = SwitchState.to_state(state)

    @property
    def is_on(self):
        return self.state