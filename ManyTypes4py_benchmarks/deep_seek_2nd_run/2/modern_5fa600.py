"""Provide pre-made queries on top of the recorder component."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from typing import Any, cast, TypedDict, Optional, Union, Dict, List, Tuple, Set, Literal

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

_FIELD_MAP = {'metadata_id': 0, 'state': 1, 'last_updated_ts': 2}

class StateResult(TypedDict, total=False):
    state: str
    last_changed: str
    last_updated: str
    attributes: Dict[str, Any]

def _stmt_and_join_attributes(
    no_attributes: bool,
    include_last_changed: bool,
    include_last_reported: bool
) -> Select[Tuple[int, str, float, ...]]:
    """Return the statement and if StateAttributes should be joined."""
    _select = select(States.metadata_id, States.state, States.last_updated_ts)
    if include_last_changed:
        _select = _select.add_columns(States.last_changed_ts)
    if include_last_reported:
        _select = _select.add_columns(States.last_reported_ts)
    if not no_attributes:
        _select = _select.add_columns(SHARED_ATTR_OR_LEGACY_ATTRIBUTES)
    return _select

def _stmt_and_join_attributes_for_start_state(
    no_attributes: bool,
    include_last_changed: bool,
    include_last_reported: bool
) -> Select[Tuple[int, str, float, ...]]:
    """Return the statement and if StateAttributes should be joined."""
    _select = select(States.metadata_id, States.state)
    _select = _select.add_columns(literal(value=0).label('last_updated_ts'))
    if include_last_changed:
        _select = _select.add_columns(literal(value=0).label('last_changed_ts'))
    if include_last_reported:
        _select = _select.add_columns(literal(value=0).label('last_reported_ts'))
    if not no_attributes:
        _select = _select.add_columns(SHARED_ATTR_OR_LEGACY_ATTRIBUTES)
    return _select

def _select_from_subquery(
    subquery: Subquery,
    no_attributes: bool,
    include_last_changed: bool,
    include_last_reported: bool
) -> Select[Tuple[int, str, float, ...]]:
    """Return the statement to select from the union."""
    base_select = select(subquery.c.metadata_id, subquery.c.state, subquery.c.last_updated_ts)
    if include_last_changed:
        base_select = base_select.add_columns(subquery.c.last_changed_ts)
    if include_last_reported:
        base_select = base_select.add_columns(subquery.c.last_reported_ts)
    if no_attributes:
        return base_select
    return base_select.add_columns(subquery.c.attributes)

def get_significant_states(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_ids: Optional[Iterable[str]] = None,
    filters: Optional[Filters] = None,
    include_start_time_state: bool = True,
    significant_changes_only: bool = True,
    minimal_response: bool = False,
    no_attributes: bool = False,
    compressed_state_format: bool = False
) -> Dict[str, List[Union[State, Dict[str, Any]]]:
    """Wrap get_significant_states_with_session with an sql session."""
    with session_scope(hass=hass, read_only=True) as session:
        return get_significant_states_with_session(
            hass, session, start_time, end_time, entity_ids, filters,
            include_start_time_state, significant_changes_only,
            minimal_response, no_attributes, compressed_state_format
        )

def _significant_states_stmt(
    start_time_ts: float,
    end_time_ts: Optional[float],
    single_metadata_id: Optional[int],
    metadata_ids: List[int],
    metadata_ids_in_significant_domains: List[int],
    significant_changes_only: bool,
    no_attributes: bool,
    include_start_time_state: bool,
    run_start_ts: Optional[float]
) -> Union[Select, CompoundSelect]:
    """Query the database for significant state changes."""
    include_last_changed = not significant_changes_only
    stmt = _stmt_and_join_attributes(no_attributes, include_last_changed, False)
    if significant_changes_only:
        if metadata_ids_in_significant_domains:
            stmt = stmt.filter(States.metadata_id.in_(metadata_ids_in_significant_domains) | (States.last_changed_ts == States.last_updated_ts) | States.last_changed_ts.is_(None))
        else:
            stmt = stmt.filter((States.last_changed_ts == States.last_updated_ts) | States.last_changed_ts.is_(None))
    stmt = stmt.filter(States.metadata_id.in_(metadata_ids)).filter(States.last_updated_ts > start_time_ts)
    if end_time_ts:
        stmt = stmt.filter(States.last_updated_ts < end_time_ts)
    if not no_attributes:
        stmt = stmt.outerjoin(StateAttributes, States.attributes_id == StateAttributes.attributes_id)
    if not include_start_time_state or not run_start_ts:
        return stmt.order_by(States.metadata_id, States.last_updated_ts)
    unioned_subquery = union_all(
        _select_from_subquery(
            _get_start_time_state_stmt(start_time_ts, single_metadata_id, metadata_ids, no_attributes, include_last_changed).subquery(),
            no_attributes, include_last_changed, False
        ),
        _select_from_subquery(stmt.subquery(), no_attributes, include_last_changed, False)
    ).subquery()
    return _select_from_subquery(unioned_subquery, no_attributes, include_last_changed, False).order_by(
        unioned_subquery.c.metadata_id, unioned_subquery.c.last_updated_ts
    )

def get_significant_states_with_session(
    hass: HomeAssistant,
    session: Session,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_ids: Optional[Iterable[str]] = None,
    filters: Optional[Filters] = None,
    include_start_time_state: bool = True,
    significant_changes_only: bool = True,
    minimal_response: bool = False,
    no_attributes: bool = False,
    compressed_state_format: bool = False
) -> Dict[str, List[Union[State, Dict[str, Any]]]]:
    """Return states changes during UTC period start_time - end_time."""
    if filters is not None:
        raise NotImplementedError('Filters are no longer supported')
    if not entity_ids:
        raise ValueError('entity_ids must be provided')
    entity_id_to_metadata_id: Dict[str, Optional[int]] = {}
    metadata_ids_in_significant_domains: List[int] = []
    instance = get_instance(hass)
    if not (entity_id_to_metadata_id := instance.states_meta_manager.get_many(entity_ids, session, False)) or not (possible_metadata_ids := extract_metadata_ids(entity_id_to_metadata_id)):
        return {}
    metadata_ids = possible_metadata_ids
    if significant_changes_only:
        metadata_ids_in_significant_domains = [
            metadata_id for entity_id, metadata_id in entity_id_to_metadata_id.items()
            if metadata_id is not None and split_entity_id(entity_id)[0] in SIGNIFICANT_DOMAINS
        ]
    oldest_ts: Optional[float] = None
    if include_start_time_state and (not (oldest_ts := _get_oldest_possible_ts(hass, start_time))):
        include_start_time_state = False
    start_time_ts = start_time.timestamp()
    end_time_ts = datetime_to_timestamp_or_none(end_time)
    single_metadata_id = metadata_ids[0] if len(metadata_ids) == 1 else None
    stmt = lambda_stmt(
        lambda: _significant_states_stmt(
            start_time_ts, end_time_ts, single_metadata_id, metadata_ids,
            metadata_ids_in_significant_domains, significant_changes_only,
            no_attributes, include_start_time_state, oldest_ts
        ),
        track_on=[
            bool(single_metadata_id), bool(metadata_ids_in_significant_domains),
            bool(end_time_ts), significant_changes_only,
            no_attributes, include_start_time_state
        ]
    )
    return _sorted_states_to_dict(
        execute_stmt_lambda_element(session, stmt, None, end_time, orm_rows=False),
        start_time_ts if include_start_time_state else None,
        entity_ids,
        entity_id_to_metadata_id,
        minimal_response,
        compressed_state_format,
        no_attributes=no_attributes
    )

def get_full_significant_states_with_session(
    hass: HomeAssistant,
    session: Session,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_ids: Optional[Iterable[str]] = None,
    filters: Optional[Filters] = None,
    include_start_time_state: bool = True,
    significant_changes_only: bool = True,
    no_attributes: bool = False
) -> Dict[str, List[State]]:
    """Variant of get_significant_states_with_session."""
    return cast(
        Dict[str, List[State]],
        get_significant_states_with_session(
            hass=hass, session=session, start_time=start_time, end_time=end_time,
            entity_ids=entity_ids, filters=filters,
            include_start_time_state=include_start_time_state,
            significant_changes_only=significant_changes_only,
            minimal_response=False, no_attributes=no_attributes
        )
    )

def _state_changed_during_period_stmt(
    start_time_ts: float,
    end_time_ts: Optional[float],
    single_metadata_id: int,
    no_attributes: bool,
    limit: Optional[int],
    include_start_time_state: bool,
    run_start_ts: Optional[float],
    include_last_reported: bool
) -> Union[Select, CompoundSelect]:
    stmt = _stmt_and_join_attributes(no_attributes, False, include_last_reported).filter(
        ((States.last_changed_ts == States.last_updated_ts) | States.last_changed_ts.is_(None)) &
        (States.last_updated_ts > start_time_ts)
    ).filter(States.metadata_id == single_metadata_id)
    if end_time_ts:
        stmt = stmt.filter(States.last_updated_ts < end_time_ts)
    if not no_attributes:
        stmt = stmt.outerjoin(StateAttributes, States.attributes_id == StateAttributes.attributes_id)
    if limit:
        stmt = stmt.limit(limit)
    stmt = stmt.order_by(States.metadata_id, States.last_updated_ts)
    if not include_start_time_state or not run_start_ts:
        return stmt
    return _select_from_subquery(
        union_all(
            _select_from_subquery(
                _get_single_entity_start_time_stmt(
                    start_time_ts, single_metadata_id, no_attributes, False, include_last_reported
                ).subquery(),
                no_attributes, False, include_last_reported
            ),
            _select_from_subquery(stmt.subquery(), no_attributes, False, include_last_reported)
        ).subquery(),
        no_attributes, False, include_last_reported
    )

def state_changes_during_period(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_id: Optional[str] = None,
    no_attributes: bool = False,
    descending: bool = False,
    limit: Optional[int] = None,
    include_start_time_state: bool = True
) -> Dict[str, List[State]]:
    """Return states changes during UTC period start_time - end_time."""
    has_last_reported = get_instance(hass).schema_version >= LAST_REPORTED_SCHEMA_VERSION
    if not entity_id:
        raise ValueError('entity_id must be provided')
    entity_ids = [entity_id.lower()]
    with session_scope(hass=hass, read_only=True) as session:
        instance = get_instance(hass)
        if not (possible_metadata_id := instance.states_meta_manager.get(entity_id, session, False)):
            return {}
        single_metadata_id = possible_metadata_id
        entity_id_to_metadata_id = {entity_id: single_metadata_id}
        oldest_ts: Optional[float] = None
        if include_start_time_state and (not (oldest_ts := _get_oldest_possible_ts(hass, start_time))):
            include_start_time_state = False
        start_time_ts = start_time.timestamp()
        end_time_ts = datetime_to_timestamp_or_none(end_time)
        stmt = lambda_stmt(
            lambda: _state_changed_during_period_stmt(
                start_time_ts, end_time_ts, single_metadata_id, no_attributes,
                limit, include_start_time_state, oldest_ts, has_last_reported
            ),
            track_on=[bool(end_time_ts), no_attributes, bool(limit), include_start_time_state, has_last_reported]
        )
        return cast(
            Dict[str, List[State]],
            _sorted_states_to_dict(
                execute_stmt_lambda_element(session, stmt, None, end_time, orm_rows=False),
                start_time_ts if include_start_time_state else None,
                entity_ids,
                entity_id_to_metadata_id,
                descending=descending,
                no_attributes=no_attributes
            )
        )

def _get_last_state_changes_single_stmt(metadata_id: int) -> Select:
    return _stmt_and_join_attributes(False, False, False).join(
        (lastest_state_for_metadata_id := select(
            States.metadata_id.label('max_metadata_id'),
            func.max(States.last_updated_ts).label('max_last_updated')
        ).filter(States.metadata_id == metadata_id).group_by(States.metadata_id).subquery()),
        and_(
            States.metadata_id == lastest_state_for_metadata_id.c.max_metadata_id,
            States.last_updated_ts == lastest_state_for_metadata_id.c.max_last_updated
        )
    ).outerjoin(StateAttributes, States.attributes_id == StateAttributes.attributes_id).order_by(States.state_id.desc())

def _get_last_state_changes_multiple_stmt(
    number_of_states: int,
    metadata_id: int,
    include_last_reported: bool
) -> Select:
    return _stmt_and_join_attributes(False, False, include_last_reported).where(
        States.state_id == select(States.state_id)
        .filter(States.metadata_id == metadata_id)
        .order_by(States.last_updated_ts.desc())
        .limit(number_of_states)
        .subquery().c.state_id
    ).outerjoin(StateAttributes, States.attributes_id == StateAttributes.attributes_id).order_by(States.state_id.desc())

def get_last_state_changes(
    hass: HomeAssistant,
    number_of_states: int,
    entity_id: str
) -> Dict[str, List[State]]:
    """Return the last number_of_states."""
    has_last_reported = get_instance(hass).schema_version >= LAST_REPORTED_SCHEMA_VERSION
    entity_id_lower = entity_id.lower()
    entity_ids = [entity_id_lower]
    with session_scope(hass=hass, read_only=True) as session:
        instance = get_instance(hass)
        if not (possible_metadata_id := instance.states_meta_manager.get(entity_id, session, False)):
            return {}
        metadata_id = possible_metadata_id
        entity_id_to_metadata_id = {entity_id_lower: metadata_id}
        if number_of_states == 1:
            stmt = lambda_stmt(lambda: _get_last_state_changes_single_stmt(metadata_id))
        else:
            stmt = lambda_stmt(
                lambda: _get_last_state_changes_multiple_stmt(number_of_states, metadata_id, has_last_reported),
                track_on=[has_last_reported]
            )
        states = list(execute_stmt_lambda_element(session, stmt, orm_rows=False))
        return cast(
            Dict[str, List[State]],
            _sorted_states_to_dict(reversed(states), None, entity_ids, entity_id_to_metadata_id, no_attributes=False)
        )

def _get_start_time_state_for_entities_stmt(
    epoch_time: float,
    metadata_ids: List[int],
    no_attributes: bool,
    include_last_changed: bool
) -> Select:
    """Baked query to get states for specific entities."""
    stmt = _stmt_and_join_attributes_for_start_state(no_attributes, include_last_changed, False).select_from(
        StatesMeta).join(
            States,
            and_(
                States.last_updated_ts == select(States.last_updated_ts)
                    .where((StatesMeta.metadata_id == States.metadata_id) & (States.last_updated_ts < epoch_time))
                    .order_by(