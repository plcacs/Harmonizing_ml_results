    def __init__(self, name: str, title: str = None, description: str = None, state: bool = SwitchState.ON) -> None:
    def serialize(self) -> dict:
    def __repr__(self) -> str:
    @classmethod
    def find_by_name(cls, name: str) -> 'Switch':
    @classmethod
    def find_all(cls) -> List['Switch']:
    def set_state(self, state: str) -> None:
    @property
    def is_on(self) -> bool:
