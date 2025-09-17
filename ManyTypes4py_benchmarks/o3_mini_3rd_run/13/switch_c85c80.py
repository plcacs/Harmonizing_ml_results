from typing import List, Optional, Dict, Any

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
    switches: List["Switch"] = []

    def __init__(self, name: str, title: Optional[str] = None, description: Optional[str] = None, state: bool = SwitchState.ON) -> None:
        self.group: str = 'switch'
        self.name: str = name
        self.title: Optional[str] = title
        self.description: Optional[str] = description
        self.state: bool = state
        Switch.switches.append(self)

    def serialize(self) -> Dict[str, Any]:
        return {
            'group': 'switch',
            'name': self.name,
            'type': 'text',
            'title': self.title,
            'description': self.description,
            'value': 'ON' if self.is_on else 'OFF'
        }

    def __repr__(self) -> str:
        return 'Switch(name={!r}, description={!r}, state={!r})'.format(
            self.name, self.description, SwitchState.to_string(self.state)
        )

    @classmethod
    def find_by_name(cls, name: str) -> Optional["Switch"]:
        for s in Switch.switches:
            if s.name == name:
                return s
        return None

    @classmethod
    def find_all(cls) -> List["Switch"]:
        return Switch.switches

    def set_state(self, state: str) -> None:
        self.state = SwitchState.to_state(state)

    @property
    def is_on(self) -> bool:
        return self.state