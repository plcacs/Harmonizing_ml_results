"""Purge old data helper."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Set
from datetime import datetime
import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, List, Dict, cast
from sqlalchemy.orm.session import Session
from homeassistant.util.collection import chunked_or_all
from .db_schema import Events, States, StatesMeta
from .models import DatabaseEngine
from .queries import (
    attributes_ids_exist_in_states, 
    attributes_ids_exist_in_states_with_fast_in_distinct, 
    data_ids_exist_in_events, 
    data_ids_exist_in_events_with_fast_in_distinct, 
    delete_event_data_rows, 
    delete_event_rows, 
    delete_event_types_rows, 
    delete_recorder_runs_rows, 
    delete_states_attributes_rows, 
    delete_states_meta_rows, 
    delete_states_rows, 
    delete_statistics_runs_rows, 
    delete_statistics_short_term_rows, 
    disconnect_states_rows, 
    find_entity_ids_to_purge, 
    find_event_types_to_purge, 
    find_events_to_purge, 
    find_latest_statistics_runs_run_id, 
    find_legacy_detached_states_and_attributes_to_purge, 
    find_legacy_event_state_and_attributes_and_data_ids_to_purge, 
    find_legacy_row, 
    find_short_term_statistics_to_purge, 
    find_states_to_purge, 
    find_statistics_runs_to_purge
)
from .repack import repack_database
from .util import retryable_database_job, session_scope

if TYPE_CHECKING:
    from . import Recorder

_LOGGER = logging.getLogger(__name__)
DEFAULT_STATES_BATCHES_PER_PURGE = 20
DEFAULT_EVENTS_BATCHES_PER_PURGE = 15

@retryable_database_job('purge')
def purge_old_data(
    instance: Recorder,
    purge_before: datetime,
    repack: bool,
    apply_filter: bool = False,
    events_batch_size: int = DEFAULT_EVENTS_BATCHES_PER_PURGE,
    states_batch_size: int = DEFAULT_STATES_BATCHES_PER_PURGE
) -> bool:
    """Purge events and states older than purge_before."""
    _LOGGER.debug('Purging states and events before target %s', purge_before.isoformat(sep=' ', timespec='seconds'))
    with session_scope(session=instance.get_session()) as session:
        has_more_to_purge = False
        if instance.use_legacy_events_index and _purging_legacy_format(session):
            _LOGGER.debug('Purge running in legacy format as there are states with event_id remaining')
            has_more_to_purge |= _purge_legacy_format(instance, session, purge_before)
        else:
            _LOGGER.debug('Purge running in new format as there are NO states with event_id remaining')
            has_more_to_purge |= _purge_states_and_attributes_ids(instance, session, states_batch_size, purge_before)
            has_more_to_purge |= _purge_events_and_data_ids(instance, session, events_batch_size, purge_before)
        statistics_runs = _select_statistics_runs_to_purge(session, purge_before, instance.max_bind_vars)
        short_term_statistics = _select_short_term_statistics_to_purge(session, purge_before, instance.max_bind_vars)
        if statistics_runs:
            _purge_statistics_runs(session, statistics_runs)
        if short_term_statistics:
            _purge_short_term_statistics(session, short_term_statistics)
        if has_more_to_purge or statistics_runs or short_term_statistics:
            _LOGGER.debug("Purging hasn't fully completed yet")
            return False
        if apply_filter and (not _purge_filtered_data(instance, session)):
            _LOGGER.debug("Cleanup filtered data hasn't fully completed yet")
            return False
        _purge_old_event_types(instance, session)
        if instance.states_meta_manager.active:
            _purge_old_entity_ids(instance, session)
        _purge_old_recorder_runs(instance, session, purge_before)
    with session_scope(session=instance.get_session(), read_only=True) as session:
        instance.recorder_runs_manager.load_from_db(session)
        instance.states_manager.load_from_db(session)
    if repack:
        repack_database(instance)
    return True

def _purging_legacy_format(session: Session) -> bool:
    """Check if there are any legacy event_id linked states rows remaining."""
    return bool(session.execute(find_legacy_row()).scalar())

def _purge_legacy_format(instance: Recorder, session: Session, purge_before: datetime) -> bool:
    """Purge rows that are still linked by the event_ids."""
    event_ids, state_ids, attributes_ids, data_ids = _select_legacy_event_state_and_attributes_and_data_ids_to_purge(session, purge_before, instance.max_bind_vars)
    _purge_state_ids(instance, session, state_ids)
    _purge_unused_attributes_ids(instance, session, attributes_ids)
    _purge_event_ids(session, event_ids)
    _purge_unused_data_ids(instance, session, data_ids)
    detached_state_ids, detached_attributes_ids = _select_legacy_detached_state_and_attributes_and_data_ids_to_purge(session, purge_before, instance.max_bind_vars)
    _purge_state_ids(instance, session, detached_state_ids)
    _purge_unused_attributes_ids(instance, session, detached_attributes_ids)
    return bool(event_ids or state_ids or attributes_ids or data_ids or detached_state_ids or detached_attributes_ids)

def _purge_states_and_attributes_ids(
    instance: Recorder,
    session: Session,
    states_batch_size: int,
    purge_before: datetime
) -> bool:
    """Purge states and linked attributes id in a batch."""
    database_engine = instance.database_engine
    assert database_engine is not None
    has_remaining_state_ids_to_purge = True
    attributes_ids_batch: Set[int] = set()
    max_bind_vars = instance.max_bind_vars
    for _ in range(states_batch_size):
        state_ids, attributes_ids = _select_state_attributes_ids_to_purge(session, purge_before, max_bind_vars)
        if not state_ids:
            has_remaining_state_ids_to_purge = False
            break
        _purge_state_ids(instance, session, state_ids)
        attributes_ids_batch = attributes_ids_batch | attributes_ids
    _purge_unused_attributes_ids(instance, session, attributes_ids_batch)
    _LOGGER.debug('After purging states and attributes_ids remaining=%s', has_remaining_state_ids_to_purge)
    return has_remaining_state_ids_to_purge

def _purge_events_and_data_ids(
    instance: Recorder,
    session: Session,
    events_batch_size: int,
    purge_before: datetime
) -> bool:
    """Purge states and linked attributes id in a batch."""
    has_remaining_event_ids_to_purge = True
    data_ids_batch: Set[int] = set()
    max_bind_vars = instance.max_bind_vars
    for _ in range(events_batch_size):
        event_ids, data_ids = _select_event_data_ids_to_purge(session, purge_before, max_bind_vars)
        if not event_ids:
            has_remaining_event_ids_to_purge = False
            break
        _purge_event_ids(session, event_ids)
        data_ids_batch = data_ids_batch | data_ids
    _purge_unused_data_ids(instance, session, data_ids_batch)
    _LOGGER.debug('After purging event and data_ids remaining=%s', has_remaining_event_ids_to_purge)
    return has_remaining_event_ids_to_purge

def _select_state_attributes_ids_to_purge(
    session: Session,
    purge_before: datetime,
    max_bind_vars: int
) -> Tuple[Set[int], Set[int]]:
    """Return sets of state and attribute ids to purge."""
    state_ids: Set[int] = set()
    attributes_ids: Set[int] = set()
    for state_id, attributes_id in session.execute(find_states_to_purge(purge_before.timestamp(), max_bind_vars)).all():
        state_ids.add(state_id)
        if attributes_id:
            attributes_ids.add(attributes_id)
    _LOGGER.debug('Selected %s state ids and %s attributes_ids to remove', len(state_ids), len(attributes_ids))
    return (state_ids, attributes_ids)

def _select_event_data_ids_to_purge(
    session: Session,
    purge_before: datetime,
    max_bind_vars: int
) -> Tuple[Set[int], Set[int]]:
    """Return sets of event and data ids to purge."""
    event_ids: Set[int] = set()
    data_ids: Set[int] = set()
    for event_id, data_id in session.execute(find_events_to_purge(purge_before.timestamp(), max_bind_vars)).all():
        event_ids.add(event_id)
        if data_id:
            data_ids.add(data_id)
    _LOGGER.debug('Selected %s event ids and %s data_ids to remove', len(event_ids), len(data_ids))
    return (event_ids, data_ids)

def _select_unused_attributes_ids(
    instance: Recorder,
    session: Session,
    attributes_ids: Set[int],
    database_engine: DatabaseEngine
) -> Set[int]:
    """Return a set of attributes ids that are not used by any states in the db."""
    if not attributes_ids:
        return set()
    seen_ids: Set[int] = set()
    if not database_engine.optimizer.slow_range_in_select:
        query = attributes_ids_exist_in_states_with_fast_in_distinct
    else:
        query = attributes_ids_exist_in_states
    for attributes_ids_chunk in chunked_or_all(attributes_ids, instance.max_bind_vars):
        seen_ids.update((state[0] for state in session.execute(query(attributes_ids_chunk)).all()))
    to_remove = attributes_ids - seen_ids
    _LOGGER.debug('Selected %s shared attributes to remove', len(to_remove))
    return to_remove

def _purge_unused_attributes_ids(instance: Recorder, session: Session, attributes_ids_batch: Set[int]) -> None:
    """Purge unused attributes ids."""
    database_engine = instance.database_engine
    assert database_engine is not None
    if (unused_attribute_ids_set := _select_unused_attributes_ids(instance, session, attributes_ids_batch, database_engine)):
        _purge_batch_attributes_ids(instance, session, unused_attribute_ids_set)

def _select_unused_event_data_ids(
    instance: Recorder,
    session: Session,
    data_ids: Set[int],
    database_engine: DatabaseEngine
) -> Set[int]:
    """Return a set of event data ids that are not used by any events in the db."""
    if not data_ids:
        return set()
    seen_ids: Set[int] = set()
    if not database_engine.optimizer.slow_range_in_select:
        query = data_ids_exist_in_events_with_fast_in_distinct
    else:
        query = data_ids_exist_in_events
    for data_ids_chunk in chunked_or_all(data_ids, instance.max_bind_vars):
        seen_ids.update((state[0] for state in session.execute(query(data_ids_chunk)).all()))
    to_remove = data_ids - seen_ids
    _LOGGER.debug('Selected %s shared event data to remove', len(to_remove))
    return to_remove

def _purge_unused_data_ids(instance: Recorder, session: Session, data_ids_batch: Set[int]) -> None:
    database_engine = instance.database_engine
    assert database_engine is not None
    if (unused_data_ids_set := _select_unused_event_data_ids(instance, session, data_ids_batch, database_engine)):
        _purge_batch_data_ids(instance, session, unused_data_ids_set)

def _select_statistics_runs_to_purge(session: Session, purge_before: datetime, max_bind_vars: int) -> List[int]:
    """Return a list of statistic runs to purge."""
    statistic_runs = session.execute(find_statistics_runs_to_purge(purge_before, max_bind_vars)).all()
    statistic_runs_list = [run_id for run_id, in statistic_runs]
    if (last_run := session.execute(find_latest_statistics_runs_run_id()).scalar()) and last_run in statistic_runs_list:
        statistic_runs_list.remove(last_run)
    _LOGGER.debug('Selected %s statistic runs to remove', len(statistic_runs))
    return statistic_runs_list

def _select_short_term_statistics_to_purge(session: Session, purge_before: datetime, max_bind_vars: int) -> List[int]:
    """Return a list of short term statistics to purge."""
    statistics = session.execute(find_short_term_statistics_to_purge(purge_before, max_bind_vars)).all()
    _LOGGER.debug('Selected %s short term statistics to remove', len(statistics))
    return [statistic_id for statistic_id, in statistics]

def _select_legacy_detached_state_and_attributes_and_data_ids_to_purge(
    session: Session,
    purge_before: datetime,
    max_bind_vars: int
) -> Tuple[Set[int], Set[int]]:
    """Return a list of state, and attribute ids to purge."""
    states = session.execute(find_legacy_detached_states_and_attributes_to_purge(purge_before.timestamp(), max_bind_vars)).all()
    _LOGGER.debug('Selected %s state ids to remove', len(states))
    state_ids: Set[int] = set()
    attributes_ids: Set[int] = set()
    for state_id, attributes_id in states:
        if state_id:
            state_ids.add(state_id)
        if attributes_id:
            attributes_ids.add(attributes_id)
    return (state_ids, attributes_ids)

def _select_legacy_event_state_and_attributes_and_data_ids_to_purge(
    session: Session,
    purge_before: datetime,
    max_bind_vars: int
) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
    """Return a list of event, state, and attribute ids to purge linked by the event_id."""
    events = session.execute(find_legacy_event_state_and_attributes_and_data_ids_to_purge(purge_before.timestamp(), max_bind_vars)).all()
    _LOGGER.debug('Selected %s event ids to remove', len(events))
    event_ids: Set[int] = set()
    state_ids: Set[int] = set()
    attributes_ids: Set[int] = set()
    data_ids: Set[int] = set()
    for event_id, data_id, state_id, attributes_id in events:
        event_ids.add(event_id)
        if state_id:
            state_ids.add(state_id)
        if attributes_id:
            attributes_ids.add(attributes_id)
        if data_id:
            data_ids.add(data_id)
    return (event_ids, state_ids, attributes_ids, data_ids)

def _purge_state_ids(instance: Recorder, session: Session, state_ids: Set[int]) -> None:
    """Disconnect states and delete by state id."""
    if not state_ids:
        return
    disconnected_rows = session.execute(disconnect_states_rows(state_ids))
    _LOGGER.debug('Updated %s states to remove old_state_id', disconnected_rows)
    deleted_rows = session.execute(delete_states_rows(state_ids))
    _LOGGER.debug('Deleted %s states', deleted_rows)
    instance.states_manager.evict_purged_state_ids(state_ids)

def _purge_batch_attributes_ids(instance: Recorder, session: Session, attributes_ids: Set[int]) -> None:
    """Delete old attributes ids in batches of max_bind_vars."""
    for attributes_ids_chunk in chunked_or_all(attributes_ids, instance.max_bind_vars):
        deleted_rows = session.execute(delete_states_attributes_rows(attributes_ids_chunk))
        _LOGGER.debug('Deleted %s attribute states', deleted_rows)
    instance.state_attributes_manager.evict_purged(attributes_ids)

def _purge_batch_data_ids(instance: Recorder, session: Session, data_ids: Set[int]) -> None:
    """Delete old event data ids in batches of max_bind_vars."""
    for data_ids_chunk in chunked_or_all(data_ids, instance.max_bind_vars):
        deleted_rows = session.execute(delete_event_data_rows(data_ids_chunk))
        _LOGGER.debug('Deleted %s data events', deleted_rows)
    instance.event_data_manager.evict_purged(data_ids)

def _purge_statistics_runs(session: Session, statistics_runs: List[int]) -> None:
    """Delete by run_id."""
    deleted_rows = session.execute(delete_statistics_runs_rows(statistics_runs))
    _LOGGER.debug('Deleted %s statistic runs', deleted_rows)

def _purge_short_term_statistics(session: Session, short_term_statistics: List[int]) -> None:
    """Delete by id."""
    deleted_rows = session.execute(delete_statistics_short_term_rows(short_term_statistics))
    _LOGGER.debug('Deleted %s short term statistics', deleted_rows)

def _purge_event_ids(session: Session, event_ids: Set[int]) -> None:
    """Delete by event id."""
    if not event_ids:
        return
    deleted_rows = session.execute(delete_event_rows(event_ids))
    _LOGGER.debug('Deleted %s events', deleted_rows)

def _purge_old_recorder_runs(instance: Recorder, session: Session, purge_before: datetime) ->