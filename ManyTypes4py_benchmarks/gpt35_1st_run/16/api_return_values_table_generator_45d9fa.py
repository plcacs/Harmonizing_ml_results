from typing import Any, List, Dict

class EventData:
    op_type: Any

class APIReturnValuesTablePreprocessor(Preprocessor):
    def __init__(self, md: markdown.Markdown, config: Dict[str, Any]) -> None:
        super().__init__(md)

    def run(self, lines: List[str]) -> List[str]:
        done: bool = False

    def render_desc(self, description: str, spacing: int, data_type: str, return_value: str = None) -> str:

    def render_oneof_block(self, object_schema: Dict[str, Any], spacing: int) -> List[str]:

    def render_table(self, return_values: Dict[str, Any], spacing: int) -> List[str]:

    def generate_event_strings(self, event_data: EventData) -> List[str]:

    def generate_events_table(self, events_by_type: Dict[str, List[str]]) -> List[str]:

    def render_events(self, events_dict: Dict[str, Any]) -> List[str]:
