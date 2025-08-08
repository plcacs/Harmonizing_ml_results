from typing import Dict, Any, Union

class StateMachine(AlarmModel):

    Severity: Dict[str, int]
    Colors: Dict[str, Union[Dict[str, str], str]]
    Status: Dict[str, str]
    DEFAULT_STATUS: str
    DEFAULT_NORMAL_SEVERITY: str
    DEFAULT_PREVIOUS_SEVERITY: str

    def register(self, app: Any) -> None:
        self.name: str = 'ANSI/ISA 18.2'
        StateMachine.Severity: Dict[str, int] = app.config.get('SEVERITY_MAP') or SEVERITY_MAP
        StateMachine.Colors: Dict[str, Union[Dict[str, str], str]] = app.config.get('COLOR_MAP') or COLOR_MAP
        StateMachine.Status: Dict[str, str] = STATUS_MAP
        StateMachine.DEFAULT_STATUS: str = A_NORM
        StateMachine.DEFAULT_NORMAL_SEVERITY: str = app.config.get('DEFAULT_NORMAL_SEVERITY') or DEFAULT_NORMAL_SEVERITY
        StateMachine.DEFAULT_PREVIOUS_SEVERITY: str = app.config.get('DEFAULT_PREVIOUS_SEVERITY') or DEFAULT_PREVIOUS_SEVERITY

    def trend(self, previous: str, current: str) -> str:
        valid_severities: List[str] = sorted(StateMachine.Severity, key=StateMachine.Severity.get)
        assert previous in StateMachine.Severity, f'Severity is not one of {', '.join(valid_severities)}'
        assert current in StateMachine.Severity, f'Severity is not one of {', '.join(valid_severities)}'
        if StateMachine.Severity[previous] < StateMachine.Severity[current]:
            return MORE_SEVERE
        elif StateMachine.Severity[previous] > StateMachine.Severity[current]:
            return LESS_SEVERE
        else:
            return NO_CHANGE

    def transition(self, alert: Any, current_status: str = None, previous_status: str = None, action: str = None, **kwargs: Any) -> Tuple[str, str]:
        state: str = current_status or StateMachine.DEFAULT_STATUS
        current_severity: str = alert.severity
        previous_severity: str = alert.previous_severity or StateMachine.DEFAULT_PREVIOUS_SEVERITY

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
        # Remaining code omitted for brevity
