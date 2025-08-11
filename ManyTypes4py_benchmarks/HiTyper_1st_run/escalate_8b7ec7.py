import logging
from alerta.app import alarm_model
from alerta.exceptions import InvalidAction
from alerta.plugins import PluginBase
LOG = logging.getLogger('alerta.plugins')
ACTION_ESCALATE = 'escalate'
escalate_map = {}

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

    def pre_receive(self, alert: list[tuple[str]], **kwargs) -> list[tuple[str]]:
        return alert

    def post_receive(self, alert: list[tuple[str]], **kwargs) -> None:
        return

    def status_change(self, alert: Union[str, set[str], typing.Sequence[str]], status: Union[str, set[str], typing.Sequence[str]], text: Union[str, set[str], typing.Sequence[str]], **kwargs) -> None:
        return

    def take_action(self, alert: str, action: Union[str, None], text: str, **kwargs) -> tuple[typing.Optional[str]]:
        if action == ACTION_ESCALATE:
            severity_level = str(alarm_model.Severity[alert.severity])
            try:
                alert.severity = escalate_map.get(severity_level)[0]
                text = 'alert severity escalated'
            except TypeError:
                raise InvalidAction(f'Can not escalate alert severity beyond "{alert.severity}".')
        return (alert, action, text)

    def take_note(self, alert: Union[str, dict, list[str]], text: Union[str, dict, list[str]], **kwargs) -> None:
        raise NotImplementedError

    def delete(self, alert: Union[list[tuple[str]], list[dict[str, typing.Any]], str], **kwargs) -> None:
        raise NotImplementedError