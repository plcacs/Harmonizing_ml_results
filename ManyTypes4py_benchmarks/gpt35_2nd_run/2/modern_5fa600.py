from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from typing import Any, cast
from sqlalchemy import CompoundSelect, Select, Subquery, and_, func, lambda_stmt, literal, select, union_all
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.session import Session
from homeassistant.const import COMPRESSED_STATE_LAST_UPDATED, COMPRESSED_STATE_STATE
from homeassistant.core import HomeAssistant, State, split_entity_id
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util
from ..const import LAST_REPORTED_SCHEMA_VERSION
from ..db_schema import SHARED_ATTR_OR_LEGACY_ATTRIBUTES, StateAttributes, States, StatesMeta
from ..filters import Filters
from ..models import LazyState, datetime_to_timestamp_or_none, extract_metadata_ids, row_to_compressed_state
from ..util import execute_stmt_lambda_element, session_scope
from .const import LAST_CHANGED_KEY, NEED_ATTRIBUTE_DOMAINS, SIGNIFICANT_DOMAINS, STATE_KEY

_FIELD_MAP: dict[str, int] = {'metadata_id': 0, 'state': 1, 'last_updated_ts': 2}

def _stmt_and_join_attributes(no_attributes: bool, include_last_changed: bool, include_last_reported: bool) -> Select:
    ...

def _stmt_and_join_attributes_for_start_state(no_attributes: bool, include_last_changed: bool, include_last_reported: bool) -> Select:
    ...

def _select_from_subquery(subquery: Subquery, no_attributes: bool, include_last_changed: bool, include_last_reported: bool) -> Select:
    ...

def get_significant_states(hass: HomeAssistant, start_time: datetime, end_time: datetime = None, entity_ids: Iterable[str] = None, filters: Filters = None, include_start_time_state: bool = True, significant_changes_only: bool = True, minimal_response: bool = False, no_attributes: bool = False, compressed_state_format: bool = False) -> dict[str, list[State]:
    ...

def _significant_states_stmt(start_time_ts: float, end_time_ts: float, single_metadata_id: int, metadata_ids: list[int], metadata_ids_in_significant_domains: list[int], significant_changes_only: bool, no_attributes: bool, include_start_time_state: bool, run_start_ts: float) -> Select:
    ...

def get_significant_states_with_session(hass: HomeAssistant, session: Session, start_time: datetime, end_time: datetime = None, entity_ids: Iterable[str] = None, filters: Filters = None, include_start_time_state: bool = True, significant_changes_only: bool = True, minimal_response: bool = False, no_attributes: bool = False, compressed_state_format: bool = False) -> dict[str, list[State]]:
    ...

def get_full_significant_states_with_session(hass: HomeAssistant, session: Session, start_time: datetime, end_time: datetime = None, entity_ids: Iterable[str] = None, filters: Filters = None, include_start_time_state: bool = True, significant_changes_only: bool = True, no_attributes: bool = False) -> dict[str, list[State]]:
    ...

def _state_changed_during_period_stmt(start_time_ts: float, end_time_ts: float, single_metadata_id: int, no_attributes: bool, limit: int, include_start_time_state: bool, run_start_ts: float, include_last_reported: bool) -> Select:
    ...

def state_changes_during_period(hass: HomeAssistant, start_time: datetime, end_time: datetime = None, entity_id: str, no_attributes: bool = False, descending: bool = False, limit: int = None, include_start_time_state: bool = True) -> dict[str, list[State]]:
    ...

def _get_last_state_changes_single_stmt(metadata_id: int) -> Select:
    ...

def _get_last_state_changes_multiple_stmt(number_of_states: int, metadata_id: int, include_last_reported: bool) -> Select:
    ...

def get_last_state_changes(hass: HomeAssistant, number_of_states: int, entity_id: str) -> dict[str, list[State]]:
    ...

def _get_start_time_state_for_entities_stmt(epoch_time: float, metadata_ids: list[int], no_attributes: bool, include_last_changed: bool) -> Select:
    ...

def _get_oldest_possible_ts(hass: HomeAssistant, utc_point_in_time: datetime) -> float:
    ...

def _get_start_time_state_stmt(epoch_time: float, single_metadata_id: int, metadata_ids: list[int], no_attributes: bool, include_last_changed: bool) -> Select:
    ...

def _get_single_entity_start_time_stmt(epoch_time: float, metadata_id: int, no_attributes: bool, include_last_changed: bool, include_last_reported: bool) -> Select:
    ...

def _sorted_states_to_dict(states: Iterable[Row], start_time_ts: float, entity_ids: list[str], entity_id_to_metadata_id: dict[str, int], minimal_response: bool = False, compressed_state_format: bool = False, descending: bool = False, no_attributes: bool = False) -> dict[str, list[State]]:
    ...
