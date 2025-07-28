from __future__ import annotations
from collections.abc import Collection
import datetime
import sqlalchemy
from sqlalchemy import lambda_stmt, select, union_all
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.lambdas import StatementLambdaElement
from sqlalchemy.sql.selectable import CTE, CompoundSelect, Select
from homeassistant.components.recorder.db_schema import (
    ENTITY_ID_IN_EVENT,
    METADATA_ID_LAST_UPDATED_INDEX_TS,
    OLD_ENTITY_ID_IN_EVENT,
    EventData,
    Events,
    EventTypes,
    States,
    StatesMeta,
)
from .common import (
    apply_events_context_hints,
    apply_states_context_hints,
    apply_states_filters,
    select_events_context_id_subquery,
    select_events_context_only,
    select_events_without_states,
    select_states,
    select_states_context_only,
)


def _select_entities_context_ids_sub_query(
    start_day: datetime.datetime,
    end_day: datetime.datetime,
    event_type_ids: Collection[int],
    states_metadata_ids: Collection[int],
    json_quoted_entity_ids: Collection[str],
) -> Select[Any]:
    union = union_all(
        select_events_context_id_subquery(start_day, end_day, event_type_ids).where(
            apply_event_entity_id_matchers(json_quoted_entity_ids)
        ),
        apply_entities_hints(
            select(States.context_id_bin)
        ).filter((States.last_updated_ts > start_day) & (States.last_updated_ts < end_day)).where(
            States.metadata_id.in_(states_metadata_ids)
        ),
    ).subquery()
    return select(union.c.context_id_bin).group_by(union.c.context_id_bin)


def _apply_entities_context_union(
    sel: Select[Any],
    start_day: datetime.datetime,
    end_day: datetime.datetime,
    event_type_ids: Collection[int],
    states_metadata_ids: Collection[int],
    json_quoted_entity_ids: Collection[str],
) -> CompoundSelect[Any]:
    entities_cte = _select_entities_context_ids_sub_query(
        start_day, end_day, event_type_ids, states_metadata_ids, json_quoted_entity_ids
    ).cte()
    return sel.union_all(
        states_select_for_entity_ids(start_day, end_day, states_metadata_ids),
        apply_events_context_hints(
            select_events_context_only().select_from(entities_cte)
            .outerjoin(Events, entities_cte.c.context_id_bin == Events.context_id_bin)
            .outerjoin(EventTypes, Events.event_type_id == EventTypes.event_type_id)
            .outerjoin(EventData, Events.data_id == EventData.data_id)
        ),
        apply_states_context_hints(
            select_states_context_only().select_from(entities_cte)
            .outerjoin(States, entities_cte.c.context_id_bin == States.context_id_bin)
            .outerjoin(StatesMeta, States.metadata_id == StatesMeta.metadata_id)
        ),
    )


def entities_stmt(
    start_day: datetime.datetime,
    end_day: datetime.datetime,
    event_type_ids: Collection[int],
    states_metadata_ids: Collection[int],
    json_quoted_entity_ids: Collection[str],
) -> StatementLambdaElement[Any]:
    return lambda_stmt(
        lambda: _apply_entities_context_union(
            select_events_without_states(start_day, end_day, event_type_ids).where(
                apply_event_entity_id_matchers(json_quoted_entity_ids)
            ),
            start_day,
            end_day,
            event_type_ids,
            states_metadata_ids,
            json_quoted_entity_ids,
        ).order_by(Events.time_fired_ts)
    )


def states_select_for_entity_ids(
    start_day: datetime.datetime,
    end_day: datetime.datetime,
    states_metadata_ids: Collection[int],
) -> Select[Any]:
    return apply_states_filters(
        apply_entities_hints(select_states()), start_day, end_day
    ).where(States.metadata_id.in_(states_metadata_ids))


def apply_event_entity_id_matchers(
    json_quoted_entity_ids: Collection[str],
) -> ColumnElement[bool]:
    return sqlalchemy.or_(
        ENTITY_ID_IN_EVENT.is_not(None)
        & sqlalchemy.cast(ENTITY_ID_IN_EVENT, sqlalchemy.Text()).in_(json_quoted_entity_ids),
        OLD_ENTITY_ID_IN_EVENT.is_not(None)
        & sqlalchemy.cast(OLD_ENTITY_ID_IN_EVENT, sqlalchemy.Text()).in_(json_quoted_entity_ids),
    )


def apply_entities_hints(sel: Select[Any]) -> Select[Any]:
    return sel.with_hint(States, f'FORCE INDEX ({METADATA_ID_LAST_UPDATED_INDEX_TS})', dialect_name='mysql').with_hint(
        States, f'FORCE INDEX ({METADATA_ID_LAST_UPDATED_INDEX_TS})', dialect_name='mariadb'
    )