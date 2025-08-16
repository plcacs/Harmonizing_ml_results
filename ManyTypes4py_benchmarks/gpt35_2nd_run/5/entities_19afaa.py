from __future__ import annotations
from collections.abc import Collection, Iterable
import sqlalchemy
from sqlalchemy import lambda_stmt, select, union_all
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.lambdas import StatementLambdaElement
from sqlalchemy.sql.selectable import CTE, CompoundSelect, Select
from homeassistant.components.recorder.db_schema import ENTITY_ID_IN_EVENT, METADATA_ID_LAST_UPDATED_INDEX_TS, OLD_ENTITY_ID_IN_EVENT, EventData, Events, EventTypes, States, StatesMeta

def _select_entities_context_ids_sub_query(start_day: str, end_day: str, event_type_ids: Collection[int], states_metadata_ids: Collection[int], json_quoted_entity_ids: Collection[str]) -> Select:
def _apply_entities_context_union(sel: Select, start_day: str, end_day: str, event_type_ids: Collection[int], states_metadata_ids: Collection[int], json_quoted_entity_ids: Collection[str]) -> CTE:
def entities_stmt(start_day: str, end_day: str, event_type_ids: Collection[int], states_metadata_ids: Collection[int], json_quoted_entity_ids: Collection[str]) -> StatementLambdaElement:
def states_select_for_entity_ids(start_day: str, end_day: str, states_metadata_ids: Collection[int]) -> Select:
def apply_event_entity_id_matchers(json_quoted_entity_ids: Collection[str]) -> ColumnElement:
def apply_entities_hints(sel: Select) -> Select:
