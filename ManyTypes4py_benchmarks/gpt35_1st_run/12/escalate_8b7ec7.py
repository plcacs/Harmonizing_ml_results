import logging
from alerta.app import alarm_model
from alerta.exceptions import InvalidAction
from alerta.plugins import PluginBase
from alerta.models.alert import Alert
from typing import Any, Dict, Tuple

LOG: logging.Logger = logging.getLogger('alerta.plugins')
ACTION_ESCALATE: str = 'escalate'
escalate_map: Dict[str, List[str]] = {}

class EscalateSeverity(PluginBase):
    def __init__(self) -> None:
        ...

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        ...

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        ...

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> None:
        ...

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        ...

    def take_note(self, alert: Alert, text: str, **kwargs: Any) -> None:
        ...

    def delete(self, alert: Alert, **kwargs: Any) -> None:
        ...
