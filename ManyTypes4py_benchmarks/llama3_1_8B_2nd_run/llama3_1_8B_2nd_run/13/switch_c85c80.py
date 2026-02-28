from typing import List

class SwitchState:
    ON: bool = True
    OFF: bool = False

    @staticmethod
    def to_state(string: str) -> bool:
        return SwitchState.ON if string == 'ON' else SwitchState.OFF

    @staticmethod
    def to_string(state: bool) -> str:
        return 'ON' if state else 'OFF'

class Switch:
    switches: list['Switch'] = []

    def __init__(self, name: str, title: str = None, description: str = None, state: SwitchState = SwitchState.ON):
        self.group: str = 'switch'
        self.name: str = name
        self.title: str = title
        self.description: str = description
        self.state: SwitchState = state
        Switch.switches.append(self)

    def serialize(self) -> dict:
        return {'group': 'switch', 'name': self.name, 'type': 'text', 'title': self.title, 'description': self.description, 'value': 'ON' if self.is_on else 'OFF'}

    def __repr__(self) -> str:
        return 'Switch(name={!r}, description={!r}, state={!r})'.format(self.name, self.description, SwitchState.to_string(self.state))

    @classmethod
    def find_by_name(cls, name: str) -> 'Switch':
        for s in Switch.switches:
            if s.name == name:
                return s
        return None

    @classmethod
    def find_all(cls) -> list['Switch']:
        return Switch.switches

    def set_state(self, state: str) -> None:
        self.state = SwitchState.to_state(state)

    @property
    def is_on(self) -> bool:
        return self.state
