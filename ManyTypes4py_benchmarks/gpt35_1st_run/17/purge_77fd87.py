from __future__ import annotations
from collections.abc import Callable
from datetime import datetime
import logging
import time
from typing import TYPE_CHECKING, Set, List, Tuple
from sqlalchemy.orm.session import Session
from homeassistant.util.collection import chunked_or_all
from .db_schema import Events, States, StatesMeta
from .models import DatabaseEngine
from .queries import attributes_ids_exist_in_states, attributes_ids_exist_in_states_with_fast_in_distinct, data_ids_exist_in_events, data_ids_exist_in_events_with_fast_in_distinct, delete_event_data_rows, delete_event_rows, delete_event_types_rows, delete_recorder_runs_rows, delete_states_attributes_rows, delete_states_meta_rows, delete_states_rows, delete_statistics_runs_rows, delete_statistics_short_term_rows, disconnect_states_rows, find_entity_ids_to_purge, find_event_types_to_purge, find_events_to_purge, find_latest_statistics_runs_run_id, find_legacy_detached_states_and_attributes_to_purge, find_legacy_event_state_and_attributes_and_data_ids_to_purge, find_legacy_row, find_short_term_statistics_to_purge, find_states_to_purge, find_statistics_runs_to_purge
from .repack import repack_database
from .util import retryable_database_job, session_scope
if TYPE_CHECKING:
    from . import Recorder

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_STATES_BATCHES_PER_PURGE: int = 20
DEFAULT_EVENTS_BATCHES_PER_PURGE: int = 15

def purge_old_data(instance: Recorder, purge_before: datetime, repack: bool, apply_filter: bool = False, events_batch_size: int = DEFAULT_EVENTS_BATCHES_PER_PURGE, states_batch_size: int = DEFAULT_STATES_BATCHES_PER_PURGE) -> bool:
    ...

def _purging_legacy_format(session: Session) -> bool:
    ...

def _purge_legacy_format(instance: Recorder, session: Session, purge_before: datetime) -> bool:
    ...

def _purge_states_and_attributes_ids(instance: Recorder, session: Session, states_batch_size: int, purge_before: datetime) -> bool:
    ...

def _purge_events_and_data_ids(instance: Recorder, session: Session, events_batch_size: int, purge_before: datetime) -> bool:
    ...

def _select_state_attributes_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int]]:
    ...

def _select_event_data_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int]]:
    ...

def _select_unused_attributes_ids(instance: Recorder, session: Session, attributes_ids: Set[int], database_engine: DatabaseEngine) -> Set[int]:
    ...

def _purge_unused_attributes_ids(instance: Recorder, session: Session, attributes_ids_batch: Set[int]) -> None:
    ...

def _select_unused_event_data_ids(instance: Recorder, session: Session, data_ids: Set[int], database_engine: DatabaseEngine) -> Set[int]:
    ...

def _purge_unused_data_ids(instance: Recorder, session: Session, data_ids_batch: Set[int]) -> None:
    ...

def _select_statistics_runs_to_purge(session: Session, purge_before: datetime, max_bind_vars: int) -> List[int]:
    ...

def _select_short_term_statistics_to_purge(session: Session, purge_before: datetime, max_bind_vars: int) -> List[int]:
    ...

def _select_legacy_detached_state_and_attributes_and_data_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int]]:
    ...

def _select_legacy_event_state_and_attributes_and_data_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
    ...

def _purge_state_ids(instance: Recorder, session: Session, state_ids: Set[int]) -> None:
    ...

def _purge_batch_attributes_ids(instance: Recorder, session: Session, attributes_ids: Set[int]) -> None:
    ...

def _purge_batch_data_ids(instance: Recorder, session: Session, data_ids: Set[int]) -> None:
    ...

def _purge_statistics_runs(session: Session, statistics_runs: List[int]) -> None:
    ...

def _purge_short_term_statistics(session: Session, short_term_statistics: List[int]) -> None:
    ...

def _purge_event_ids(session: Session, event_ids: Set[int]) -> None:
    ...

def _purge_old_recorder_runs(instance: Recorder, session: Session, purge_before: datetime) -> None:
    ...

def _purge_old_event_types(instance: Recorder, session: Session) -> None:
    ...

def _purge_old_entity_ids(instance: Recorder, session: Session) -> None:
    ...

def _purge_filtered_data(instance: Recorder, session: Session) -> bool:
    ...

def _purge_filtered_states(instance: Recorder, session: Session, metadata_ids_to_purge: List[int], database_engine: DatabaseEngine, purge_before_timestamp: float) -> bool:
    ...

def _purge_filtered_events(instance: Recorder, session: Session, excluded_event_type_ids: List[int], purge_before_timestamp: float) -> bool:
    ...

def purge_entity_data(instance: Recorder, entity_filter: Callable, purge_before: datetime) -> bool:
    ...
