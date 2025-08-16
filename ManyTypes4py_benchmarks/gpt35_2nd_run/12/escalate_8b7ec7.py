import logging
from alerta.app import alarm_model
from alerta.exceptions import InvalidAction
from alerta.plugins import PluginBase
from typing import Dict, Any

LOG: logging.Logger = logging.getLogger('alerta.plugins')
ACTION_ESCALATE: str = 'escalate'
escalate_map: Dict[str, Any] = {}

class EscalateSeverity(PluginBase):
    def __init__(self) -> None:
        ...

    def pre_receive(self, alert, **kwargs) -> Any:
        ...

    def post_receive(self, alert, **kwargs) -> None:
        ...

    def status_change(self, alert, status, text, **kwargs) -> None:
        ...

    def take_action(self, alert, action, text, **kwargs) -> Tuple[Any, str, str]:
        ...

    def take_note(self, alert, text, **kwargs) -> None:
        ...

    def delete(self, alert, **kwargs) -> None:
        ...
