from flask import current_app
from alerta.exceptions import ApiError, InvalidAction
from alerta.models.alarms import AlarmModel
from alerta.models.enums import Action, Severity, Status, TrendIndication
from typing import Dict, List, Tuple

SEVERITY_MAP: Dict[Severity, int] = {Severity.Security: 0, Severity.Critical: 1, Severity.Major: 2, Severity.Minor: 3, Severity.Warning: 4, Severity.Indeterminate: 5, Severity.Informational: 6, Severity.Normal: 7, Severity.Ok: 7, Severity.Cleared: 7, Severity.Debug: 8, Severity.Trace: 9, Severity.Unknown: 10}
DEFAULT_NORMAL_SEVERITY: Severity = Severity.Normal
DEFAULT_INFORM_SEVERITY: Severity = Severity.Informational
DEFAULT_PREVIOUS_SEVERITY: Severity = Severity.Indeterminate
COLOR_MAP: Dict[str, Dict[Severity, str]] = {'severity': {Severity.Security: 'blue', Severity.Critical: 'red', Severity.Major: 'orange', Severity.Minor: 'yellow', Severity.Warning: 'dodgerblue', Severity.Indeterminate: 'lightblue', Severity.Cleared: '#00CC00', Severity.Normal: '#00CC00', Severity.Ok: '#00CC00', Severity.Informational: '#00CC00', Severity.Debug: '#9D006D', Severity.Trace: '#7554BF', Severity.Unknown: 'silver'}, 'status': {Status.Ack: 'skyblue', Status.Shelved: 'skyblue'}, 'text': 'black'}
STATUS_MAP: Dict[Status, str] = {Status.Open: 'A', Status.Assign: 'B', Status.Ack: 'C', Status.Shelved: 'D', Status.Blackout: 'E', Status.Closed: 'F', Status.Expired: 'G', Status.Unknown: 'H'}
ACTION_ALL: List[Action] = [Action.OPEN, Action.ASSIGN, Action.ACK, Action.UNACK, Action.SHELVE, Action.UNSHELVE, Action.CLOSE, Action.EXPIRED, Action.TIMEOUT]

class StateMachine(AlarmModel):

    @property
    def valid_severities(self) -> List[Severity]:
        return sorted(StateMachine.Severity, key=StateMachine.Severity.get)

    def register(self, app) -> None:
        from alerta.management.views import __version__
        self.name: str = f'Alerta {__version__}'
        StateMachine.Severity: Dict[Severity, int] = app.config['SEVERITY_MAP'] or SEVERITY_MAP
        StateMachine.Colors: Dict[str, Dict[Severity, str]] = app.config['COLOR_MAP'] or COLOR_MAP
        StateMachine.Status: Dict[Status, str] = STATUS_MAP
        StateMachine.DEFAULT_STATUS: Status = Status.Open
        StateMachine.DEFAULT_NORMAL_SEVERITY: Severity = app.config['DEFAULT_NORMAL_SEVERITY'] or DEFAULT_NORMAL_SEVERITY
        StateMachine.DEFAULT_INFORM_SEVERITY: Severity = app.config['DEFAULT_INFORM_SEVERITY'] or DEFAULT_INFORM_SEVERITY
        StateMachine.DEFAULT_PREVIOUS_SEVERITY: Severity = app.config['DEFAULT_PREVIOUS_SEVERITY'] or DEFAULT_PREVIOUS_SEVERITY
        if StateMachine.DEFAULT_NORMAL_SEVERITY not in StateMachine.Severity:
            raise RuntimeError(f'DEFAULT_NORMAL_SEVERITY ({StateMachine.DEFAULT_NORMAL_SEVERITY}) is not one of {", ".join(self.valid_severities)}')
        if StateMachine.DEFAULT_PREVIOUS_SEVERITY not in StateMachine.Severity:
            raise RuntimeError(f'DEFAULT_PREVIOUS_SEVERITY ({StateMachine.DEFAULT_PREVIOUS_SEVERITY}) is not one of {", ".join(self.valid_severities)}')
        StateMachine.NORMAL_SEVERITY_LEVEL: int = StateMachine.Severity[StateMachine.DEFAULT_NORMAL_SEVERITY]

    def trend(self, previous: Severity, current: Severity) -> TrendIndication:
        if previous not in StateMachine.Severity or current not in StateMachine.Severity:
            return TrendIndication.No_Change
        if StateMachine.Severity[previous] > StateMachine.Severity[current]:
            return TrendIndication.More_Severe
        elif StateMachine.Severity[previous] < StateMachine.Severity[current]:
            return TrendIndication.Less_Severe
        else:
            return TrendIndication.No_Change

    def transition(self, alert, current_status: Status = None, previous_status: Status = None, action: Action = None, **kwargs) -> Tuple[Severity, Status]:
        current_status = current_status or StateMachine.DEFAULT_STATUS
        previous_status = previous_status or StateMachine.DEFAULT_STATUS
        current_severity = alert.severity
        previous_severity = alert.previous_severity or StateMachine.DEFAULT_PREVIOUS_SEVERITY
        valid_severities: List[Severity] = sorted(StateMachine.Severity, key=StateMachine.Severity.get)
        if current_severity not in StateMachine.Severity:
            raise ApiError(f'Severity ({current_severity}) is not one of {", ".join(valid_severities)}', 400)

        def next_state(rule, severity, status) -> Tuple[Severity, Status]:
            current_app.logger.info(f'State Transition: Rule #{rule} STATE={current_status} ACTION={action or ""} SET={alert.status} SEVERITY={previous_severity}-> {current_severity} HISTORY={previous_status}-> {current_status} => SEVERITY={severity}, STATUS={status}')
            return (severity, status)
        
        # Rest of the method omitted for brevity

    @staticmethod
    def is_suppressed(alert) -> bool:
        return alert.status == Status.Blackout
