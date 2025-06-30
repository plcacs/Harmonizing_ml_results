import logging
from typing import Dict, List, Tuple, Any, Optional
from alerta.app import alarm_model
from alerta.exceptions import InvalidAction
from alerta.plugins import PluginBase
from alerta.models.alert import Alert

LOG = logging.getLogger('alerta.plugins')
ACTION_ESCALATE = 'escalate'
escalate_map: Dict[str, List[str]] = {}

class EscalateSeverity(PluginBase):
    """
    Add "escalate" custom action to web UI and CLI and increase the severity
    an alert to the next highest severity based on the configured alarm model.

    Must add "escalate" to the list of enabled 'PLUGINS' and "escalate" to
    the list of valid custom 'ACTIONS' in server settings.
    """

    def __init__(self) -> None:
        for sev in alarm_model.Severity:
            level = str(alarm_model.Severity[sev] + 1)
            if level not in escalate_map:
                escalate_map[level] = [sev]
            else:
                escalate_map[level].append(sev)
        super().__init__()

    def pre_receive(self, alert: Alert, **kwargs: Any) -> Alert:
        return alert

    def post_receive(self, alert: Alert, **kwargs: Any) -> None:
        return

    def status_change(self, alert: Alert, status: str, text: str, **kwargs: Any) -> None:
        return

    def take_action(self, alert: Alert, action: str, text: str, **kwargs: Any) -> Tuple[Alert, str, str]:
        if action == ACTION_ESCALATE:
            severity_level = str(alarm_model.Severity[alert.severity])
            try:
                alert.severity = escalate_map.get(severity_level)[0]
                text = 'alert severity escalated'
            except TypeError:
                raise InvalidAction(f'Can not escalate alert severity beyond "{alert.severity}".')
        return (alert, action, text)

    def take_note(self, alert: Alert, text: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def delete(self, alert: Alert, **kwargs: Any) -> None:
        raise NotImplementedError
