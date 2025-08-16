from __future__ import annotations
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import datetime as dt
import logging
import time
from typing import TYPE_CHECKING, Any
from sqlalchemy.engine import Result
from sqlalchemy.engine.row import Row
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.filters import Filters
from homeassistant.components.recorder.models import bytes_to_uuid_hex_or_none, extract_event_type_ids, extract_metadata_ids, process_timestamp_to_utc_isoformat
from homeassistant.components.recorder.util import execute_stmt_lambda_element, session_scope
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.const import ATTR_DOMAIN, ATTR_ENTITY_ID, ATTR_FRIENDLY_NAME, ATTR_NAME, ATTR_SERVICE, EVENT_CALL_SERVICE, EVENT_LOGBOOK_ENTRY
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from homeassistant.util.event_type import EventType
from .const import ATTR_MESSAGE, CONTEXT_DOMAIN, CONTEXT_ENTITY_ID, CONTEXT_ENTITY_ID_NAME, CONTEXT_EVENT_TYPE, CONTEXT_MESSAGE, CONTEXT_NAME, CONTEXT_SERVICE, CONTEXT_SOURCE, CONTEXT_STATE, CONTEXT_USER_ID, DOMAIN, LOGBOOK_ENTRY_DOMAIN, LOGBOOK_ENTRY_ENTITY_ID, LOGBOOK_ENTRY_ICON, LOGBOOK_ENTRY_MESSAGE, LOGBOOK_ENTRY_NAME, LOGBOOK_ENTRY_SOURCE, LOGBOOK_ENTRY_STATE, LOGBOOK_ENTRY_WHEN
from .helpers import is_sensor_continuous
from .models import CONTEXT_ID_BIN_POS, CONTEXT_ONLY_POS, CONTEXT_PARENT_ID_BIN_POS, CONTEXT_POS, CONTEXT_USER_ID_BIN_POS, ENTITY_ID_POS, EVENT_TYPE_POS, ICON_POS, ROW_ID_POS, STATE_POS, TIME_FIRED_TS_POS, EventAsRow, LazyEventPartialState, LogbookConfig, async_event_to_row
from .queries import statement_for_request
from .queries.common import PSEUDO_EVENT_STATE_CHANGED
_LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class LogbookRun:
    memoize_new_contexts: bool = True

class EventProcessor:
    def __init__(self, hass: HomeAssistant, event_types: Sequence[EventType], entity_ids: Sequence[str] = None, device_ids: Sequence[str] = None, context_id: Any = None, timestamp: bool = False, include_entity_name: bool = True):
        def switch_to_live(self) -> None:
        def get_events(self, start_day: dt, end_day: dt) -> list:
        def humanify(self, rows: Sequence[Row]) -> list:

def _humanify(hass: HomeAssistant, rows: Sequence[Row], ent_reg: er, logbook_run: LogbookRun, context_augmenter: ContextAugmenter) -> Generator:

class ContextAugmenter:
    def __init__(self, logbook_run: LogbookRun) -> None:
    def get_context(self, context_id_bin: Any, row: Row) -> Row:
    def augment(self, data: dict, context_row: Row) -> None:

def _rows_ids_match(row: Row, other_row: Row) -> bool:

class EntityNameCache:
    def __init__(self, hass: HomeAssistant) -> None:
    def get(self, entity_id: str) -> str:

class EventCache:
    def __init__(self, event_data_cache: dict) -> None:
    def get(self, row: Row) -> LazyEventPartialState:
    def clear(self) -> None:
