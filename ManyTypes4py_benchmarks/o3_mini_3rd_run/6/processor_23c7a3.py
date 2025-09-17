from __future__ import annotations
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import datetime as dt
import logging
import time
from typing import Any, Dict, Optional
from sqlalchemy.engine import Result
from sqlalchemy.engine.row import Row

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.filters import Filters
from homeassistant.components.recorder.models import (
    bytes_to_uuid_hex_or_none,
    extract_event_type_ids,
    extract_metadata_ids,
    process_timestamp_to_utc_isoformat,
)
from homeassistant.components.recorder.util import execute_stmt_lambda_element, session_scope
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.const import (
    ATTR_DOMAIN,
    ATTR_ENTITY_ID,
    ATTR_FRIENDLY_NAME,
    ATTR_NAME,
    ATTR_SERVICE,
    EVENT_CALL_SERVICE,
    EVENT_LOGBOOK_ENTRY,
)
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from homeassistant.util.event_type import EventType

from .const import (
    ATTR_MESSAGE,
    CONTEXT_DOMAIN,
    CONTEXT_ENTITY_ID,
    CONTEXT_ENTITY_ID_NAME,
    CONTEXT_EVENT_TYPE,
    CONTEXT_MESSAGE,
    CONTEXT_NAME,
    CONTEXT_SERVICE,
    CONTEXT_SOURCE,
    CONTEXT_STATE,
    CONTEXT_USER_ID,
    DOMAIN,
    LOGBOOK_ENTRY_DOMAIN,
    LOGBOOK_ENTRY_ENTITY_ID,
    LOGBOOK_ENTRY_ICON,
    LOGBOOK_ENTRY_MESSAGE,
    LOGBOOK_ENTRY_NAME,
    LOGBOOK_ENTRY_SOURCE,
    LOGBOOK_ENTRY_STATE,
    LOGBOOK_ENTRY_WHEN,
)
from .helpers import is_sensor_continuous
from .models import (
    CONTEXT_ID_BIN_POS,
    CONTEXT_ONLY_POS,
    CONTEXT_PARENT_ID_BIN_POS,
    CONTEXT_POS,
    CONTEXT_USER_ID_BIN_POS,
    ENTITY_ID_POS,
    EVENT_TYPE_POS,
    ICON_POS,
    ROW_ID_POS,
    STATE_POS,
    TIME_FIRED_TS_POS,
    EventAsRow,
    LazyEventPartialState,
    LogbookConfig,
    async_event_to_row,
)
from .queries import statement_for_request
from .queries.common import PSEUDO_EVENT_STATE_CHANGED

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LogbookRun:
    context_lookup: Dict[Any, Any]
    external_events: Dict[Any, tuple[str, Callable[[Any], dict[str, Any]]]]
    event_cache: EventCache
    entity_name_cache: EntityNameCache
    include_entity_name: bool
    timestamp: bool
    memoize_new_contexts: bool = True


class EventProcessor:
    def __init__(
        self,
        hass: HomeAssistant,
        event_types: Sequence[str],
        entity_ids: Optional[Sequence[str]] = None,
        device_ids: Optional[Sequence[str]] = None,
        context_id: Optional[str] = None,
        timestamp: bool = False,
        include_entity_name: bool = True,
    ) -> None:
        assert not (context_id and (entity_ids or device_ids)), "can't pass in both context_id and (entity_ids or device_ids)"
        self.hass: HomeAssistant = hass
        self.ent_reg: er.EntityRegistry = er.async_get(hass)
        self.event_types: Sequence[str] = event_types
        self.entity_ids: Optional[Sequence[str]] = entity_ids
        self.device_ids: Optional[Sequence[str]] = device_ids
        self.context_id: Optional[str] = context_id
        logbook_config: LogbookConfig = hass.data[DOMAIN]
        self.filters: Filters = logbook_config.sqlalchemy_filter
        self.logbook_run: LogbookRun = LogbookRun(
            context_lookup={None: None},
            external_events=logbook_config.external_events,
            event_cache=EventCache({}),
            entity_name_cache=EntityNameCache(self.hass),
            include_entity_name=include_entity_name,
            timestamp=timestamp,
        )
        self.context_augmenter: ContextAugmenter = ContextAugmenter(self.logbook_run)

    @property
    def limited_select(self) -> bool:
        return bool(self.entity_ids or self.context_id or self.device_ids)

    def switch_to_live(self) -> None:
        self.logbook_run.event_cache.clear()
        self.logbook_run.context_lookup.clear()
        self.logbook_run.memoize_new_contexts = False

    def get_events(self, start_day: dt, end_day: dt) -> list[dict[str, Any]]:
        with session_scope(hass=self.hass, read_only=True) as session:
            metadata_ids: Optional[Any] = None
            instance = get_instance(self.hass)
            if self.entity_ids:
                metadata_ids = extract_metadata_ids(instance.states_meta_manager.get_many(self.entity_ids, session, False))
            event_type_ids: tuple[Any, ...] = tuple(extract_event_type_ids(instance.event_type_manager.get_many(self.event_types, session)))
            stmt = statement_for_request(start_day, end_day, event_type_ids, self.entity_ids, metadata_ids, self.device_ids, self.filters, self.context_id)
            rows = execute_stmt_lambda_element(session, stmt, orm_rows=False)
            return self.humanify(rows)

    def humanify(self, rows: Sequence[Any]) -> list[dict[str, Any]]:
        return list(_humanify(self.hass, rows, self.ent_reg, self.logbook_run, self.context_augmenter))


def _humanify(
    hass: HomeAssistant,
    rows: Sequence[Any],
    ent_reg: er.EntityRegistry,
    logbook_run: LogbookRun,
    context_augmenter: ContextAugmenter,
) -> Generator[dict[str, Any], None, None]:
    continuous_sensors: Dict[str, bool] = {}
    context_lookup: Dict[Any, Any] = logbook_run.context_lookup
    external_events: Dict[Any, tuple[str, Callable[[Any], dict[str, Any]]]] = logbook_run.external_events
    event_cache_get: Callable[[Any], LazyEventPartialState] = logbook_run.event_cache.get
    entity_name_cache_get: Callable[[str], str] = logbook_run.entity_name_cache.get
    include_entity_name: bool = logbook_run.include_entity_name
    timestamp: bool = logbook_run.timestamp
    memoize_new_contexts: bool = logbook_run.memoize_new_contexts
    get_context: Callable[[Any, Any], Optional[Any]] = context_augmenter.get_context
    for row in rows:
        context_id_bin: Any = row[CONTEXT_ID_BIN_POS]
        if memoize_new_contexts and context_id_bin not in context_lookup:
            context_lookup[context_id_bin] = row
        if row[CONTEXT_ONLY_POS]:
            continue
        event_type: Any = row[EVENT_TYPE_POS]
        if event_type == EVENT_CALL_SERVICE:
            continue
        if event_type is PSEUDO_EVENT_STATE_CHANGED:
            entity_id: str = row[ENTITY_ID_POS]
            if entity_id is None:
                continue
            is_continuous: Optional[bool] = continuous_sensors.get(entity_id)
            if is_continuous is None and split_entity_id(entity_id)[0] == SENSOR_DOMAIN:
                is_continuous = is_sensor_continuous(hass, ent_reg, entity_id)
                continuous_sensors[entity_id] = is_continuous
            if is_continuous:
                continue
            data: dict[str, Any] = {LOGBOOK_ENTRY_STATE: row[STATE_POS], LOGBOOK_ENTRY_ENTITY_ID: entity_id}
            if include_entity_name:
                data[LOGBOOK_ENTRY_NAME] = entity_name_cache_get(entity_id)
            if (icon := row[ICON_POS]):
                data[LOGBOOK_ENTRY_ICON] = icon
        elif event_type in external_events:
            domain, describe_event = external_events[event_type]
            try:
                data = describe_event(event_cache_get(row))
            except Exception:
                _LOGGER.exception('Error with %s describe event for %s', domain, event_type)
                continue
            data[LOGBOOK_ENTRY_DOMAIN] = domain
        elif event_type == EVENT_LOGBOOK_ENTRY:
            event = event_cache_get(row)
            if not (event_data := event.data):
                continue
            entry_domain: Optional[str] = event_data.get(ATTR_DOMAIN)
            entry_entity_id: Optional[Any] = event_data.get(ATTR_ENTITY_ID)
            if entry_domain is None and entry_entity_id is not None:
                entry_domain = split_entity_id(str(entry_entity_id))[0]
            data = {
                LOGBOOK_ENTRY_NAME: event_data.get(ATTR_NAME),
                LOGBOOK_ENTRY_MESSAGE: event_data.get(ATTR_MESSAGE),
                LOGBOOK_ENTRY_DOMAIN: entry_domain,
                LOGBOOK_ENTRY_ENTITY_ID: entry_entity_id,
            }
        else:
            continue
        time_fired_ts: float = row[TIME_FIRED_TS_POS]
        if timestamp:
            when: float = time_fired_ts or time.time()
        else:
            ts_dt = dt_util.utc_from_timestamp(time_fired_ts) or dt_util.utcnow()
            when = process_timestamp_to_utc_isoformat(ts_dt)
        data[LOGBOOK_ENTRY_WHEN] = when
        if (context_user_id_bin := row[CONTEXT_USER_ID_BIN_POS]):
            data[CONTEXT_USER_ID] = bytes_to_uuid_hex_or_none(context_user_id_bin)
        context_row: Optional[Any] = get_context(context_id_bin, row)
        if context_row and not ((row is context_row or _rows_ids_match(row, context_row)) and (not (context_parent := row[CONTEXT_PARENT_ID_BIN_POS])
            or not (context_row := get_context(context_parent, context_row))
            or row is context_row
            or _rows_ids_match(row, context_row))):
            context_augmenter.augment(data, context_row)
        yield data


class ContextAugmenter:
    def __init__(self, logbook_run: LogbookRun) -> None:
        self.context_lookup: Dict[Any, Any] = logbook_run.context_lookup
        self.entity_name_cache: EntityNameCache = logbook_run.entity_name_cache
        self.external_events: Dict[Any, tuple[str, Callable[[Any], dict[str, Any]]]] = logbook_run.external_events
        self.event_cache: EventCache = logbook_run.event_cache
        self.include_entity_name: bool = logbook_run.include_entity_name

    def get_context(self, context_id_bin: Any, row: Any) -> Optional[Any]:
        if context_id_bin is not None and (context_row := self.context_lookup.get(context_id_bin)):
            return context_row
        if type(row) is EventAsRow and (context := row[CONTEXT_POS]) is not None and ((origin_event := context.origin_event) is not None):
            return async_event_to_row(origin_event)
        return None

    def augment(self, data: dict[str, Any], context_row: Any) -> None:
        event_type: Any = context_row[EVENT_TYPE_POS]
        if (context_entity_id := context_row[ENTITY_ID_POS]):
            data[CONTEXT_STATE] = context_row[STATE_POS]
            data[CONTEXT_ENTITY_ID] = context_entity_id
            if self.include_entity_name:
                data[CONTEXT_ENTITY_ID_NAME] = self.entity_name_cache.get(context_entity_id)
            return
        if event_type == EVENT_CALL_SERVICE:
            event = self.event_cache.get(context_row)
            event_data = event.data
            data[CONTEXT_DOMAIN] = event_data.get(ATTR_DOMAIN)
            data[CONTEXT_SERVICE] = event_data.get(ATTR_SERVICE)
            data[CONTEXT_EVENT_TYPE] = event_type
            return
        if event_type not in self.external_events:
            return
        domain, describe_event = self.external_events[event_type]
        data[CONTEXT_EVENT_TYPE] = event_type
        data[CONTEXT_DOMAIN] = domain
        event = self.event_cache.get(context_row)
        try:
            described = describe_event(event)
        except Exception:
            _LOGGER.exception('Error with %s describe event for %s', domain, event_type)
            return
        if (name := described.get(LOGBOOK_ENTRY_NAME)):
            data[CONTEXT_NAME] = name
        if (message := described.get(LOGBOOK_ENTRY_MESSAGE)):
            data[CONTEXT_MESSAGE] = message
        if (source := described.get(LOGBOOK_ENTRY_SOURCE)):
            data[CONTEXT_SOURCE] = source
        if not (attr_entity_id := described.get(LOGBOOK_ENTRY_ENTITY_ID)):
            return
        data[CONTEXT_ENTITY_ID] = attr_entity_id
        if self.include_entity_name:
            data[CONTEXT_ENTITY_ID_NAME] = self.entity_name_cache.get(attr_entity_id)


def _rows_ids_match(row: Any, other_row: Any) -> bool:
    return bool((row_id := row[ROW_ID_POS]) and row_id == other_row[ROW_ID_POS])


class EntityNameCache:
    def __init__(self, hass: HomeAssistant) -> None:
        self._hass: HomeAssistant = hass
        self._names: Dict[str, str] = {}

    def get(self, entity_id: str) -> str:
        if entity_id in self._names:
            return self._names[entity_id]
        current_state = self._hass.states.get(entity_id)
        if current_state and (friendly_name := current_state.attributes.get(ATTR_FRIENDLY_NAME)):
            self._names[entity_id] = friendly_name
        else:
            return split_entity_id(entity_id)[1].replace('_', ' ')
        return self._names[entity_id]


class EventCache:
    def __init__(self, event_data_cache: Dict[Any, Any]) -> None:
        self._event_data_cache: Dict[Any, Any] = event_data_cache
        self.event_cache: Dict[Any, LazyEventPartialState] = {}

    def get(self, row: Any) -> LazyEventPartialState:
        if type(row) is EventAsRow:
            return LazyEventPartialState(row, self._event_data_cache)
        if (event := self.event_cache.get(row)):
            return event
        self.event_cache[row] = lazy_event = LazyEventPartialState(row, self._event_data_cache)
        return lazy_event

    def clear(self) -> None:
        self._event_data_cache = {}
        self.event_cache = {}