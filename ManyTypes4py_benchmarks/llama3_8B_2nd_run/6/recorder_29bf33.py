from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import suppress
import datetime
import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

def _get_sensor_states(hass: HomeAssistant) -> List[State]:
    ...

def _time_weighted_average(fstates: Iterable[Tuple[Any, State]], start: datetime.datetime, end: datetime.datetime) -> float:
    ...

def _get_units(fstates: Iterable[Tuple[Any, State]]) -> Set[str]:
    ...

def _equivalent_units(units: Set[str]) -> bool:
    ...

def _entity_history_to_float_and_state(entity_history: Iterable[State]) -> List[Tuple[float, State]]:
    ...

def _is_numeric(state: State) -> bool:
    ...

def _normalize_states(hass: HomeAssistant, old_metadatas: Dict[str, Tuple[str, Dict]], fstates: Iterable[Tuple[Any, State]], entity_id: str) -> Tuple[str, List[Tuple[Any, State]]]:
    ...

def _suggest_report_issue(hass: HomeAssistant, entity_id: str) -> str:
    ...

def warn_dip(hass: HomeAssistant, entity_id: str, state: State, previous_fstate: Optional[float]) -> None:
    ...

def warn_negative(hass: HomeAssistant, entity_id: str, state: State) -> None:
    ...

def reset_detected(hass: HomeAssistant, entity_id: str, fstate: float, previous_fstate: Optional[float], state: State) -> bool:
    ...

def _wanted_statistics(sensor_states: Iterable[State]) -> Dict[str, Set[str]]:
    ...

def _last_reset_as_utc_isoformat(last_reset_s: Optional[str], entity_id: str) -> Optional[str]:
    ...

def _timestamp_to_isoformat_or_none(timestamp: Optional[int]) -> Optional[str]:
    ...

def compile_statistics(hass: HomeAssistant, session: Session, start: datetime.datetime, end: datetime.datetime) -> statistics.PlatformCompiledStatistics:
    ...

def list_statistic_ids(hass: HomeAssistant, statistic_ids: Optional[Set[str]], statistic_type: Optional[str]) -> Dict[str, Dict]:
    ...

@callback
def _update_issues(report_issue: Callable[[str, str, Dict], None], sensor_states: Iterable[State], metadatas: Dict[str, Tuple[str, Dict]]) -> None:
    ...

def update_statistics_issues(hass: HomeAssistant, session: Session) -> None:
    ...

def validate_statistics(hass: HomeAssistant) -> Dict[str, List[statistics.ValidationIssue]]:
    ...
