"""
Alarm states and transition paths.

See ANSI/ISA 18.2 Management of Alarm Systems for the Process Industries
https://www.isa.org/store/ansi/isa-182-2016/46962105

"""
from typing import Dict, Tuple, Optional, Any
from flask import current_app
from alerta.models.alarms import AlarmModel

CRITICAL: str = 'Critical'
HIGH: str = 'High'
MEDIUM: str = 'Medium'
LOW: str = 'Low'
ADVISORY: str = 'Advisory'
OK: str = 'OK'
UNKNOWN: str = 'Unknown'
SEVERITY_MAP: Dict[str, int] = {CRITICAL: 5, HIGH: 4, MEDIUM: 3, LOW: 2, ADVISORY: 1, OK: 0}
DEFAULT_NORMAL_SEVERITY: str = OK
DEFAULT_PREVIOUS_SEVERITY: str = OK
COLOR_MAP: Dict[str, Dict[str, str]] = {
    'severity': {
        CRITICAL: 'red',
        HIGH: 'orange',
        MEDIUM: 'yellow',
        LOW: 'dodgerblue',
        ADVISORY: 'lightblue',
        OK: '#00CC00',
        UNKNOWN: 'silver'
    },
    'text': 'black'
}
A_NORM: str = 'NORM'
B_UNACK: str = 'UNACK'
C_ACKED: str = 'ACKED'
D_RTNUN: str = 'RTNUN'
E_SHLVD: str = 'SHLVD'
F_DSUPR: str = 'DSUPR'
G_OOSRV: str = 'OOSRV'
STATUS_MAP: Dict[str, str] = {A_NORM: 'A', B_UNACK: 'B', C_ACKED: 'C', D_RTNUN: 'D', E_SHLVD: 'E', F_DSUPR: 'F', G_OOSRV: 'G'}
MORE_SEVERE: str = 'moreSevere'
NO_CHANGE: str = 'noChange'
LESS_SEVERE: str = 'lessSevere'
ACTION_ACK: str = 'ack'
ACTION_UNACK: str = 'unack'
ACTION_SHELVE: str = 'shelve'
ACTION_UNSHELVE: str = 'unshelve'

class StateMachine(AlarmModel):

    def register(self, app: Any) -> None:
        self.name = 'ANSI/ISA 18.2'
        StateMachine.Severity = app.config['SEVERITY_MAP'] or SEVERITY_MAP
        StateMachine.Colors = app.config['COLOR_MAP'] or COLOR_MAP
        StateMachine.Status = STATUS_MAP
        StateMachine.DEFAULT_STATUS = A_NORM
        StateMachine.DEFAULT_NORMAL_SEVERITY = app.config['DEFAULT_NORMAL_SEVERITY'] or DEFAULT_NORMAL_SEVERITY
        StateMachine.DEFAULT_PREVIOUS_SEVERITY = app.config['DEFAULT_PREVIOUS_SEVERITY'] or DEFAULT_PREVIOUS_SEVERITY

    def trend(self, previous: str, current: str) -> str:
        valid_severities = sorted(StateMachine.Severity, key=StateMachine.Severity.get)
        assert previous in StateMachine.Severity, f'Severity is not one of {", ".join(valid_severities)}'
        assert current in StateMachine.Severity, f'Severity is not one of {", ".join(valid_severities)}'
        if StateMachine.Severity[previous] < StateMachine.Severity[current]:
            return MORE_SEVERE
        elif StateMachine.Severity[previous] > StateMachine.Severity[current]:
            return LESS_SEVERE
        else:
            return NO_CHANGE

    def transition(self, alert: Any, current_status: Optional[str] = None, previous_status: Optional[str] = None, action: Optional[str] = None, **kwargs: Any) -> Tuple[str, str]:
        state = current_status or StateMachine.DEFAULT_STATUS
        current_severity = alert.severity
        previous_severity = alert.previous_severity or StateMachine.DEFAULT_PREVIOUS_SEVERITY

        def next_state(rule: str, severity: str, status: str) -> Tuple[str, str]:
            current_app.logger.info('State Transition: Rule {}: STATE={} => SEVERITY={}, STATUS={}'.format(rule, state, severity, status))
            return (severity, status)
        if not action and alert.status != StateMachine.DEFAULT_STATUS:
            return next_state('External State Change, Any (*) -> Any (*)', current_severity, alert.status)
        if action == ACTION_SHELVE:
            return next_state('Operator Shelve, Any (*) -> Shelve (E)', current_severity, E_SHLVD)
        if action == ACTION_UNSHELVE:
            if current_severity == StateMachine.DEFAULT_NORMAL_SEVERITY:
                return next_state('Operator Unshelve, Shelve (E) -> Normal (A)', current_severity, A_NORM)
            else:
                return next_state('Operator Unshelve, Shelve (E) -> Unack (B)', current_severity, B_UNACK)
        if state == A_NORM:
            if current_severity != StateMachine.DEFAULT_NORMAL_SEVERITY:
                return next_state('Alarm Occurs, Normal (A) -> Unack (B)', current_severity, B_UNACK)
        if state == B_UNACK:
            if action == ACTION_ACK:
                return next_state('Operator Ack, Unack (B) -> Ack (C)', current_severity, C_ACKED)
        if state == C_ACKED:
            if action == ACTION_UNACK:
                return next_state('Operator Unack, Ack (C) -> Unack (B)', current_severity, B_UNACK)
        if state == C_ACKED:
            if self.trend(previous_severity, current_severity) == MORE_SEVERE:
                if previous_severity != StateMachine.DEFAULT_PREVIOUS_SEVERITY:
                    return next_state('Re-Alarm, Ack (C) -> Unack (B)', current_severity, B_UNACK)
        if state == C_ACKED:
            if current_severity == StateMachine.DEFAULT_NORMAL_SEVERITY:
                return next_state('Process RTN Alarm Clears, Ack (C) -> Normal (A)', current_severity, A_NORM)
        if state == B_UNACK:
            if current_severity == StateMachine.DEFAULT_NORMAL_SEVERITY:
                return next_state('Process RTN and Alarm Clears, Unack (B) -> RTN Unack (D)', current_severity, D_RTNUN)
        if state == D_RTNUN:
            if action == ACTION_ACK:
                return next_state(' Operator Ack, RTN Unack (D) -> Normal (A)', current_severity, A_NORM)
        if state == D_RTNUN:
            if current_severity != StateMachine.DEFAULT_NORMAL_SEVERITY:
                return next_state('Re-Alarm Unack, RTN Unack (D) -> Unack (B)', current_severity, B_UNACK)
        if state == F_DSUPR:
            if current_severity == StateMachine.DEFAULT_NORMAL_SEVERITY:
                return next_state('Return from Suppressed-by-design, Suppressed-by-design (G) -> Normal (A)', current_severity, A_NORM)
            else:
                return next_state('Return from Suppressed-by-design, Suppressed-by-design (G) -> Unack (B)', current_severity, B_UNACK)
        if state == G_OOSRV:
            if current_severity == StateMachine.DEFAULT_NORMAL_SEVERITY:
                return next_state('Return from Out-of-service, Out-of-service (G) -> Normal (A)', current_severity, A_NORM)
            else:
                return next_state('Return from Out-of-service, Out-of-service (G) -> Unack (B)', current_severity, B_UNACK)
        return next_state('NOOP', current_severity, current_status)

    @staticmethod
    def is_suppressed(alert: Any) -> bool:
        return alert.status in [F_DSUPR, G_OOSRV]
