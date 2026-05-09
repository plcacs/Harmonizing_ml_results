from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from itertools import groupby
from operator import attrgetter
import time
from typing import Any, cast

def _lambda_stmt_and_join_attributes(no_attributes: bool, include_last_changed: bool) -> tuple[Callable, bool]:
    ...

def get_significant_states(
    hass: HomeAssistant, 
    start_time: datetime, 
    end_time: datetime | None = None, 
    entity_ids: Iterable[str] | None = None, 
    filters: None = None, 
    include_start_time_state: bool = True, 
    significant_changes_only: bool = True, 
    minimal_response: bool = False, 
    no_attributes: bool = False, 
    compressed_state_format: bool = False
) -> dict[str, list[State]]:
    ...

def _significant_states_stmt(
    start_time: datetime, 
    end_time: datetime | None, 
    entity_ids: Iterable[str], 
    significant_changes_only: bool, 
    no_attributes: bool
) -> Callable:
    ...

def get_significant_states_with_session(
    hass: HomeAssistant, 
    session: Session, 
    start_time: datetime, 
    end_time: datetime | None = None, 
    entity_ids: Iterable[str] | None = None, 
    filters: None = None, 
    include_start_time_state: bool = True, 
    significant_changes_only: bool = True, 
    minimal_response: bool = False, 
    no_attributes: bool = False, 
    compressed_state_format: bool = False
) -> dict[str, list[State]]:
    ...

def _state_changed_during_period_stmt(
    start_time: datetime, 
    end_time: datetime | None, 
    entity_id: str, 
    no_attributes: bool, 
    descending: bool, 
    limit: int | None
) -> Callable:
    ...

def state_changes_during_period(
    hass: HomeAssistant, 
    start_time: datetime, 
    end_time: datetime | None = None, 
    entity_id: str | None = None, 
    no_attributes: bool = False, 
    descending: bool = False, 
    limit: int | None = None, 
    include_start_time_state: bool = True
) -> dict[str, list[State]]:
    ...

def _get_last_state_changes_stmt(
    number_of_states: int, 
    entity_id: str
) -> Callable:
    ...

def get_last_state_changes(
    hass: HomeAssistant, 
    number_of_states: int, 
    entity_id: str
) -> dict[str, list[State]]:
    ...

def _get_states_for_entities_stmt(
    run_start_ts: float, 
    utc_point_in_time: datetime, 
    entity_ids: Iterable[str], 
    no_attributes: bool
) -> Callable:
    ...

def _get_rows_with_session(
    hass: HomeAssistant, 
    session: Session, 
    utc_point_in_time: datetime, 
    entity_ids: Iterable[str], 
    *, 
    no_attributes: bool = False
) -> list[Row]:
    ...

def _get_single_entity_states_stmt(
    utc_point_in_time: datetime, 
    entity_id: str, 
    no_attributes: bool = False
) -> Callable:
    ...

def _sorted_states_to_dict(
    hass: HomeAssistant, 
    session: Session, 
    states: Iterable[Row], 
    start_time: datetime, 
    entity_ids: Iterable[str], 
    include_start_time_state: bool = True, 
    minimal_response: bool = False, 
    no_attributes: bool = False, 
    compressed_state_format: bool = False
) -> dict[str, list[State]]:
    ...
