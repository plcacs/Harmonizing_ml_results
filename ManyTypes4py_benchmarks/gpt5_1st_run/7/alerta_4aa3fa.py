from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict

from flask import Flask, current_app
from alerta.exceptions import ApiError, InvalidAction
from alerta.models.alarms import AlarmModel
from alerta.models.enums import Action, Severity, Status, TrendIndication


class ColorMapType(TypedDict):
    severity: Dict[Severity, str]
    status: Dict[Status, str]
    text: str


class AlertLike(Protocol):
    severity: Severity
    previous_severity: Optional[Severity]
    status: Status


SEVERITY_MAP: Dict[Severity, int] = {
    Severity.Security: 0,
    Severity.Critical: 1,
    Severity.Major: 2,
    Severity.Minor: 3,
    Severity.Warning: 4,
    Severity.Indeterminate: 5,
    Severity.Informational: 6,
    Severity.Normal: 7,
    Severity.Ok: 7,
    Severity.Cleared: 7,
    Severity.Debug: 8,
    Severity.Trace: 9,
    Severity.Unknown: 10,
}
DEFAULT_NORMAL_SEVERITY: Severity = Severity.Normal
DEFAULT_INFORM_SEVERITY: Severity = Severity.Informational
DEFAULT_PREVIOUS_SEVERITY: Severity = Severity.Indeterminate
COLOR_MAP: ColorMapType = {
    "severity": {
        Severity.Security: "blue",
        Severity.Critical: "red",
        Severity.Major: "orange",
        Severity.Minor: "yellow",
        Severity.Warning: "dodgerblue",
        Severity.Indeterminate: "lightblue",
        Severity.Cleared: "#00CC00",
        Severity.Normal: "#00CC00",
        Severity.Ok: "#00CC00",
        Severity.Informational: "#00CC00",
        Severity.Debug: "#9D006D",
        Severity.Trace: "#7554BF",
        Severity.Unknown: "silver",
    },
    "status": {Status.Ack: "skyblue", Status.Shelved: "skyblue"},
    "text": "black",
}
STATUS_MAP: Dict[Status, str] = {
    Status.Open: "A",
    Status.Assign: "B",
    Status.Ack: "C",
    Status.Shelved: "D",
    Status.Blackout: "E",
    Status.Closed: "F",
    Status.Expired: "G",
    Status.Unknown: "H",
}
ACTION_ALL: List[Action] = [
    Action.OPEN,
    Action.ASSIGN,
    Action.ACK,
    Action.UNACK,
    Action.SHELVE,
    Action.UNSHELVE,
    Action.CLOSE,
    Action.EXPIRED,
    Action.TIMEOUT,
]


class StateMachine(AlarmModel):
    Severity: Dict[Severity, int]
    Colors: ColorMapType
    Status: Dict[Status, str]
    DEFAULT_STATUS: Status
    DEFAULT_NORMAL_SEVERITY: Severity
    DEFAULT_INFORM_SEVERITY: Severity
    DEFAULT_PREVIOUS_SEVERITY: Severity
    NORMAL_SEVERITY_LEVEL: int

    @property
    def valid_severities(self) -> List[Severity]:
        return sorted(StateMachine.Severity, key=StateMachine.Severity.get)

    def register(self, app: Flask) -> None:
        from alerta.management.views import __version__

        self.name = f'Alerta {__version__}'
        StateMachine.Severity = app.config["SEVERITY_MAP"] or SEVERITY_MAP
        StateMachine.Colors = app.config["COLOR_MAP"] or COLOR_MAP
        StateMachine.Status = STATUS_MAP
        StateMachine.DEFAULT_STATUS = Status.Open
        StateMachine.DEFAULT_NORMAL_SEVERITY = (
            app.config["DEFAULT_NORMAL_SEVERITY"] or DEFAULT_NORMAL_SEVERITY
        )
        StateMachine.DEFAULT_INFORM_SEVERITY = (
            app.config["DEFAULT_INFORM_SEVERITY"] or DEFAULT_INFORM_SEVERITY
        )
        StateMachine.DEFAULT_PREVIOUS_SEVERITY = (
            app.config["DEFAULT_PREVIOUS_SEVERITY"] or DEFAULT_PREVIOUS_SEVERITY
        )
        if StateMachine.DEFAULT_NORMAL_SEVERITY not in StateMachine.Severity:
            raise RuntimeError(
                "DEFAULT_NORMAL_SEVERITY ({}) is not one of {}".format(
                    StateMachine.DEFAULT_NORMAL_SEVERITY, ", ".join(self.valid_severities)
                )
            )
        if StateMachine.DEFAULT_PREVIOUS_SEVERITY not in StateMachine.Severity:
            raise RuntimeError(
                "DEFAULT_PREVIOUS_SEVERITY ({}) is not one of {}".format(
                    StateMachine.DEFAULT_PREVIOUS_SEVERITY,
                    ", ".join(self.valid_severities),
                )
            )
        StateMachine.NORMAL_SEVERITY_LEVEL = StateMachine.Severity[
            StateMachine.DEFAULT_NORMAL_SEVERITY
        ]

    def trend(self, previous: Severity, current: Severity) -> TrendIndication:
        if previous not in StateMachine.Severity or current not in StateMachine.Severity:
            return TrendIndication.No_Change
        if StateMachine.Severity[previous] > StateMachine.Severity[current]:
            return TrendIndication.More_Severe
        elif StateMachine.Severity[previous] < StateMachine.Severity[current]:
            return TrendIndication.Less_Severe
        else:
            return TrendIndication.No_Change

    def transition(
        self,
        alert: AlertLike,
        current_status: Optional[Status] = None,
        previous_status: Optional[Status] = None,
        action: Optional[Action] = None,
        **kwargs: Any,
    ) -> Tuple[Severity, Status]:
        current_status = current_status or StateMachine.DEFAULT_STATUS
        previous_status = previous_status or StateMachine.DEFAULT_STATUS
        current_severity: Severity = alert.severity
        previous_severity: Severity = (
            alert.previous_severity or StateMachine.DEFAULT_PREVIOUS_SEVERITY
        )
        valid_severities: List[Severity] = sorted(
            StateMachine.Severity, key=StateMachine.Severity.get
        )
        if current_severity not in StateMachine.Severity:
            raise ApiError(
                f"Severity ({current_severity}) is not one of {', '.join(valid_severities)}",
                400,
            )

        def next_state(rule: str, severity: Severity, status: Status) -> Tuple[Severity, Status]:
            current_app.logger.info(
                "State Transition: Rule #{} STATE={:8s} ACTION={:8s} SET={:8s} SEVERITY={:13s}-> {:8s} HISTORY={:8s}-> {:8s} => SEVERITY={:8s}, STATUS={:8s}".format(
                    rule,
                    current_status,
                    action or "",
                    alert.status,
                    previous_severity,
                    current_severity,
                    previous_status,
                    current_status,
                    severity,
                    status,
                )
            )
            return (severity, status)

        if action and action not in ACTION_ALL:
            return next_state("ACT-1", current_severity, alert.status)

        if not action and alert.status != StateMachine.DEFAULT_STATUS:
            if StateMachine.Severity[current_severity] == StateMachine.NORMAL_SEVERITY_LEVEL:
                return next_state("SET-1", StateMachine.DEFAULT_NORMAL_SEVERITY, Status.Closed)
            return next_state("SET-*", current_severity, alert.status)

        state: Status = current_status

        if action == Action.UNACK:
            if state == Status.Ack:
                return next_state("UNACK-1", current_severity, previous_status)
            else:
                raise InvalidAction(f"invalid action for current {state} status")

        if action == Action.UNSHELVE:
            if state == Status.Shelved:
                return next_state("UNSHL-1", current_severity, previous_status)
            else:
                raise InvalidAction(f"invalid action for current {state} status")

        if action == Action.EXPIRED:
            return next_state("EXP-0", current_severity, Status.Expired)

        if action == Action.TIMEOUT:
            if previous_status == Status.Ack:
                return next_state("ACK-0", current_severity, Status.Ack)
            else:
                return next_state("OPEN-0", current_severity, Status.Open)

        if state == Status.Open:
            if action == Action.OPEN:
                raise InvalidAction(f"alert is already in {state} status")
            if action == Action.ACK:
                return next_state("OPEN-1", current_severity, Status.Ack)
            if action == Action.SHELVE:
                return next_state("OPEN-2", current_severity, Status.Shelved)
            if action == Action.CLOSE:
                return next_state("OPEN-3", StateMachine.DEFAULT_NORMAL_SEVERITY, Status.Closed)

        if state == Status.Assign:
            pass

        if state == Status.Ack:
            if action == Action.OPEN:
                return next_state("ACK-1", current_severity, Status.Open)
            if action == Action.ACK:
                raise InvalidAction(f"alert is already in {state} status")
            if action == Action.SHELVE:
                return next_state("ACK-2", current_severity, Status.Shelved)
            if action == Action.CLOSE:
                return next_state("ACK-3", StateMachine.DEFAULT_NORMAL_SEVERITY, Status.Closed)
            if previous_severity != StateMachine.DEFAULT_PREVIOUS_SEVERITY:
                if self.trend(previous_severity, current_severity) == TrendIndication.More_Severe:
                    return next_state("ACK-4", current_severity, Status.Open)

        if state == Status.Shelved:
            if action == Action.OPEN:
                return next_state("SHL-1", current_severity, Status.Open)
            if action == Action.ACK:
                raise InvalidAction(f"invalid action for current {state} status")
            if action == Action.SHELVE:
                raise InvalidAction(f"alert is already in {state} status")
            if action == Action.CLOSE:
                return next_state("SHL-2", StateMachine.DEFAULT_NORMAL_SEVERITY, Status.Closed)

        if state == Status.Blackout:
            if action == Action.CLOSE:
                return next_state("BLK-1", StateMachine.DEFAULT_NORMAL_SEVERITY, Status.Closed)
            if previous_status != Status.Blackout:
                return next_state("BLK-2", current_severity, previous_status)
            else:
                return next_state("BLK-*", current_severity, alert.status)

        if state == Status.Closed:
            if action == Action.OPEN:
                return next_state("CLS-1", previous_severity, Status.Open)
            if action == Action.ACK:
                raise InvalidAction(f"invalid action for current {state} status")
            if action == Action.SHELVE:
                raise InvalidAction(f"invalid action for current {state} status")
            if action == Action.CLOSE:
                raise InvalidAction(f"alert is already in {state} status")
            if StateMachine.Severity[current_severity] != StateMachine.NORMAL_SEVERITY_LEVEL:
                if previous_status == Status.Shelved:
                    return next_state("CLS-2", previous_severity, Status.Shelved)
                else:
                    return next_state("CLS-3", previous_severity, Status.Open)

        if StateMachine.Severity[current_severity] == StateMachine.NORMAL_SEVERITY_LEVEL:
            return next_state("CLS-*", StateMachine.DEFAULT_NORMAL_SEVERITY, Status.Closed)

        if state == Status.Expired:
            if action and action != Action.OPEN:
                raise InvalidAction(f"invalid action for current {state} status")
            if StateMachine.Severity[current_severity] != StateMachine.NORMAL_SEVERITY_LEVEL:
                return next_state("EXP-1", current_severity, Status.Open)

        return next_state("ALL-*", current_severity, current_status)

    @staticmethod
    def is_suppressed(alert: AlertLike) -> bool:
        return alert.status == Status.Blackout