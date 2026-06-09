from typing import Any

# === Third-party dependency: freezegun ===
# Used symbols: freeze_time

# === Internal dependency: homeassistant.components.recorder ===
from .const import DOMAIN
from .util import get_instance

# === Internal dependency: homeassistant.components.recorder.const ===
class SupportedDialect(StrEnum): ...

# === Internal dependency: homeassistant.components.recorder.db_schema ===
class Events(Base): ...
class EventTypes(Base):
    ...
class States(Base):
class StateAttributes(Base):
class StatesMeta(Base):
class StatisticsShortTerm(Base, StatisticsBase): ...
class RecorderRuns(Base): ...
class StatisticsRuns(Base): ...

# === Internal dependency: homeassistant.components.recorder.history ===
def get_significant_states(hass, start_time, end_time=..., entity_ids=..., filters=..., include_start_time_state=..., significant_changes_only=..., minimal_response=..., no_attributes=..., compressed_state_format=...): ...

# === Internal dependency: homeassistant.components.recorder.purge ===
def purge_old_data(instance, purge_before, repack, apply_filter=..., events_batch_size=..., states_batch_size=...): ...

# === Internal dependency: homeassistant.components.recorder.queries ===
def select_event_type_ids(event_types): ...

# === Internal dependency: homeassistant.components.recorder.services ===
SERVICE_PURGE = 'purge'
SERVICE_PURGE_ENTITIES = 'purge_entities'

# === Internal dependency: homeassistant.components.recorder.tasks ===
class PurgeTask(RecorderTask): ...

# === Internal dependency: homeassistant.components.recorder.util ===
def session_scope(*, hass=..., session=..., exception_filter=..., read_only=...): ...

# === Internal dependency: homeassistant.const ===
EVENT_STATE_CHANGED = EventType(...)
EVENT_THEMES_UPDATED = 'themes_updated'
STATE_ON = 'on'

# === Internal dependency: homeassistant.helpers.typing ===
ConfigType: Any

# === Internal dependency: homeassistant.util.dt ===
def utc_to_timestamp(utc_dt): ...
UTC = dt.UTC
utcnow = partial(...)

# === Internal dependency: homeassistant.util.event_type ===
class EventType(Generic[_DataT]): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Third-party dependency: sqlalchemy.exc ===
class DatabaseError(DBAPIError):
    ...
class OperationalError(DatabaseError):

# === Internal dependency: tests.components.recorder.common ===
async def async_wait_recording_done(hass): ...
async def async_wait_purge_done(hass, max_number=...): ...
async def async_recorder_block_till_done(hass): ...
def convert_pending_states_to_meta(instance, session): ...
def convert_pending_events_to_event_types(instance, session): ...

# === Third-party dependency: voluptuous.error ===
class MultipleInvalid(Invalid): ...