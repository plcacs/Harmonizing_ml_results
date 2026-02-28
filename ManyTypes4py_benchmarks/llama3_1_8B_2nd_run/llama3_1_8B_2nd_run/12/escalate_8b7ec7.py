import logging
from alerta.app import alarm_model
from alerta.exceptions import InvalidAction
from alerta.plugins import PluginBase
LOG = logging.getLogger('alerta.plugins')
ACTION_ESCALATE = 'escalate'
escalate_map: dict[str, list[str]] = {}

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

    def pre_receive(self, alert: alarm_model.Alert, **kwargs) -> alarm_model.Alert:
        return alert

    def post_receive(self, alert: alarm_model.Alert, **kwargs) -> None:
        return

    def status_change(self, alert: alarm_model.Alert, status: str, text: str, **kwargs) -> None:
        return

    def take_action(self, alert: alarm_model.Alert, action: str, text: str, **kwargs) -> tuple[alarm_model.Alert, str, str]:
        if action == ACTION_ESCALATE:
            severity_level = str(alarm_model.Severity[alert.severity])
            try:
                alert.severity = escalate_map.get(severity_level)[0]
                text = 'alert severity escalated'
            except TypeError:
                raise InvalidAction(f'Can not escalate alert severity beyond "{alert.severity}".')
        return (alert, action, text)

    def take_note(self, alert: alarm_model.Alert, text: str, **kwargs) -> None:
        raise NotImplementedError

    def delete(self, alert: alarm_model.Alert, **kwargs) -> None:
        raise NotImplementedError
