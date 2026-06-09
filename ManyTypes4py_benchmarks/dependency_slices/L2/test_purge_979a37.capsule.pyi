from typing import Any

# === Third-party dependency: freezegun ===
# Used symbols: freeze_time

# === Internal dependency: homeassistant.components.recorder ===
# re-export: from .const import DOMAIN
# re-export: from .util import get_instance

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
def get_significant_states(hass: HomeAssistant, start_time: datetime, end_time: datetime | None = ..., entity_ids: list[str] | None = ..., filters: Filters | None = ..., include_start_time_state: bool = ..., significant_changes_only: bool = ..., minimal_response: bool = ..., no_attributes: bool = ..., compressed_state_format: bool = ...) -> dict[str, list[State | dict[str, Any]]]: ...

# === Internal dependency: homeassistant.components.recorder.purge ===
def purge_old_data(instance: Recorder, purge_before: datetime, repack: bool, apply_filter: bool = ..., events_batch_size: int = ..., states_batch_size: int = ...) -> bool: ...

# === Internal dependency: homeassistant.components.recorder.queries ===
def select_event_type_ids(event_types: tuple[str, ...]) -> Select: ...

# === Internal dependency: homeassistant.components.recorder.services ===
SERVICE_PURGE: str
SERVICE_PURGE_ENTITIES: str

# === Internal dependency: homeassistant.components.recorder.tasks ===
class PurgeTask(RecorderTask): ...

# === Internal dependency: homeassistant.components.recorder.util ===
def session_scope(*, hass: HomeAssistant | None = ..., session: Session | None = ..., exception_filter: Callable[[Exception], bool] | None = ..., read_only: bool = ...) -> Generator[Session]: ...

# === Internal dependency: homeassistant.const ===
EVENT_STATE_CHANGED: EventType[EventStateChangedData]
EVENT_THEMES_UPDATED: Final
STATE_ON: Final

# === Internal dependency: homeassistant.helpers.typing ===
ConfigType: Any

# === Internal dependency: homeassistant.util.dt ===
def utc_to_timestamp(utc_dt: dt.datetime) -> float: ...
utcnow: partial

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Third-party dependency: sqlalchemy.exc ===
class DatabaseError(DBAPIError):
    ...
class OperationalError(DatabaseError):

# === Internal dependency: tests.components.recorder.common ===
async def async_wait_recording_done(hass: HomeAssistant) -> None: ...
async def async_wait_purge_done(hass: HomeAssistant, max_number: int | None = ...) -> None: ...
async def async_recorder_block_till_done(hass: HomeAssistant) -> None: ...
def convert_pending_states_to_meta(instance: Recorder, session: Session) -> None: ...
def convert_pending_events_to_event_types(instance: Recorder, session: Session) -> None: ...

# === Third-party dependency: voluptuous.error ===
class MultipleInvalid(Invalid): ...