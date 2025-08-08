from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from itertools import groupby
from operator import attrgetter
import time
from typing import Any, cast
from sqlalchemy import Column, Text, and_, func, lambda_stmt, or_, select
from sqlalchemy.engine.row import Row
from sqlalchemy.orm.properties import MappedColumn
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.expression import literal
from sqlalchemy.sql.lambdas import StatementLambdaElement
from homeassistant.const import COMPRESSED_STATE_LAST_UPDATED, COMPRESSED_STATE_STATE
from homeassistant.core import HomeAssistant, State, split_entity_id
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util
from ..db_schema import StateAttributes, States
from ..filters import Filters
from ..models import process_timestamp_to_utc_isoformat
from ..models.legacy import LegacyLazyState, legacy_row_to_compressed_state
from ..util import execute_stmt_lambda_element, session_scope
from .const import LAST_CHANGED_KEY, NEED_ATTRIBUTE_DOMAINS, SIGNIFICANT_DOMAINS, SIGNIFICANT_DOMAINS_ENTITY_ID_LIKE, STATE_KEY

_BASE_STATES: tuple[Column, Column, Column, Column] = (States.entity_id, States.state, States.last_changed_ts, States.last_updated_ts)
_BASE_STATES_NO_LAST_CHANGED: tuple[Column, Column, Column, Column] = (States.entity_id, States.state, literal(value=None).label('last_changed_ts'), States.last_updated_ts)
_QUERY_STATE_NO_ATTR: tuple[Column, Column, Column, Column] = (*_BASE_STATES, literal(value=None, type_=Text).label('attributes'), literal(value=None, type_=Text).label('shared_attrs'))
_QUERY_STATE_NO_ATTR_NO_LAST_CHANGED: tuple[Column, Column, Column, Column] = (*_BASE_STATES_NO_LAST_CHANGED, literal(value=None, type_=Text).label('attributes'), literal(value=None, type_=Text).label('shared_attrs'))
_BASE_STATES_PRE_SCHEMA_31: tuple[Column, Column, Column, Column] = (States.entity_id, States.state, States.last_changed, States.last_updated)
_BASE_STATES_NO_LAST_CHANGED_PRE_SCHEMA_31: tuple[Column, Column, Column, Column] = (States.entity_id, States.state, literal(value=None, type_=Text).label('last_changed'), States.last_updated)
_QUERY_STATE_NO_ATTR_PRE_SCHEMA_31: tuple[Column, Column, Column, Column] = (*_BASE_STATES_PRE_SCHEMA_31, literal(value=None, type_=Text).label('attributes'), literal(value=None, type_=Text).label('shared_attrs'))
_QUERY_STATE_NO_ATTR_NO_LAST_CHANGED_PRE_SCHEMA_31: tuple[Column, Column, Column, Column] = (*_BASE_STATES_NO_LAST_CHANGED_PRE_SCHEMA_31, literal(value=None, type_=Text).label('attributes'), literal(value=None, type_=Text).label('shared_attrs'))
_QUERY_STATES_PRE_SCHEMA_25: tuple[Column, Column, Column, Column, Column] = (*_BASE_STATES_PRE_SCHEMA_31, States.attributes, literal(value=None, type_=Text).label('shared_attrs'))
_QUERY_STATES_PRE_SCHEMA_25_NO_LAST_CHANGED: tuple[Column, Column, Column, Column, Column] = (*_BASE_STATES_NO_LAST_CHANGED_PRE_SCHEMA_31, States.attributes, literal(value=None, type_=Text).label('shared_attrs'))
_QUERY_STATES_PRE_SCHEMA_31: tuple[Column, Column, Column, Column, Column] = (*_BASE_STATES_PRE_SCHEMA_31, States.attributes, StateAttributes.shared_attrs)
_QUERY_STATES_NO_LAST_CHANGED_PRE_SCHEMA_31: tuple[Column, Column, Column, Column, Column] = (*_BASE_STATES_NO_LAST_CHANGED_PRE_SCHEMA_31, States.attributes, StateAttributes.shared_attrs)
_QUERY_STATES: tuple[Column, Column, Column, Column, Column] = (*_BASE_STATES, States.attributes, StateAttributes.shared_attrs)
_QUERY_STATES_NO_LAST_CHANGED: tuple[Column, Column, Column, Column, Column] = (*_BASE_STATES_NO_LAST_CHANGED, States.attributes, StateAttributes.shared_attrs)
_FIELD_MAP: dict[str, int] = {cast(MappedColumn, field).name: idx for idx, field in enumerate(_QUERY_STATE_NO_ATTR)}
_FIELD_MAP_PRE_SCHEMA_31: dict[str, int] = {cast(MappedColumn, field).name: idx for idx, field in enumerate(_QUERY_STATES_PRE_SCHEMA_31)}

def _lambda_stmt_and_join_attributes(no_attributes: bool, include_last_changed: bool = True) -> tuple[StatementLambdaElement, bool]:
    ...

def get_significant_states(hass: HomeAssistant, start_time: datetime, end_time: datetime = None, entity_ids: Iterable[str] = None, filters: Any = None, include_start_time_state: bool = True, significant_changes_only: bool = True, minimal_response: bool = False, no_attributes: bool = False, compressed_state_format: bool = False) -> dict[str, list[State]]:
    ...

def _significant_states_stmt(start_time: datetime, end_time: datetime, entity_ids: Iterable[str], significant_changes_only: bool, no_attributes: bool) -> StatementLambdaElement:
    ...

def get_significant_states_with_session(hass: HomeAssistant, session: Session, start_time: datetime, end_time: datetime = None, entity_ids: Iterable[str] = None, filters: Any = None, include_start_time_state: bool = True, significant_changes_only: bool = True, minimal_response: bool = False, no_attributes: bool = False, compressed_state_format: bool = False) -> dict[str, list[State]]:
    ...

def get_full_significant_states_with_session(hass: HomeAssistant, session: Session, start_time: datetime, end_time: datetime = None, entity_ids: Iterable[str] = None, filters: Any = None, include_start_time_state: bool = True, significant_changes_only: bool = True, no_attributes: bool = False) -> dict[str, list[State]]:
    ...

def _state_changed_during_period_stmt(start_time: datetime, end_time: datetime, entity_id: str, no_attributes: bool, descending: bool, limit: int) -> StatementLambdaElement:
    ...

def state_changes_during_period(hass: HomeAssistant, start_time: datetime, end_time: datetime = None, entity_id: str = None, no_attributes: bool = False, descending: bool = False, limit: int = None, include_start_time_state: bool = True) -> dict[str, list[State]]:
    ...

def _get_last_state_changes_stmt(number_of_states: int, entity_id: str) -> StatementLambdaElement:
    ...

def get_last_state_changes(hass: HomeAssistant, number_of_states: int, entity_id: str) -> dict[str, list[State]]:
    ...

def _get_states_for_entities_stmt(run_start_ts: float, utc_point_in_time: datetime, entity_ids: Iterable[str], no_attributes: bool) -> StatementLambdaElement:
    ...

def _get_rows_with_session(hass: HomeAssistant, session: Session, utc_point_in_time: datetime, entity_ids: Iterable[str], *, no_attributes: bool = False) -> list[Row]:
    ...

def _get_single_entity_states_stmt(utc_point_in_time: datetime, entity_id: str, no_attributes: bool = False) -> StatementLambdaElement:
    ...

def _sorted_states_to_dict(hass: HomeAssistant, session: Session, states: Iterable[Row], start_time: datetime, entity_ids: Iterable[str], include_start_time_state: bool = True, minimal_response: bool = False, no_attributes: bool = False, compressed_state_format: bool = False) -> dict[str, list[State]]:
    ...
