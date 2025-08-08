    def __init__(self, labels: list[str]):
    def serialize_capability_resources(self) -> dict:
    def serialize_configuration(self) -> dict:
    def serialize_labels(self, resources: list[str]) -> dict:
    def __init__(self, labels: list[str], ordered: bool = False):
    def add_mode(self, value: str, labels: list[str]):
    def serialize_configuration(self) -> dict:
    def __init__(self, labels: list[str], min_value: Any, max_value: Any, precision: Any, unit: str = None):
    def add_preset(self, value: Any, labels: list[str]):
    def serialize_configuration(self) -> dict:
    def _add_action_mapping(self, semantics: dict):
    def _add_state_mapping(self, semantics: dict):
    def add_states_to_value(self, states: list[str], value: Any):
    def add_states_to_range(self, states: list[str], min_value: Any, max_value: Any):
    def add_action_to_directive(self, actions: list[str], directive: str, payload: Any):
    def serialize_semantics(self) -> dict:
