from typing import Any

# === Third-party dependency: freezegun ===
# Used symbols: freeze_time

# === Internal dependency: homeassistant.components.alexa.smart_home ===
# re-export: from .const import EVENT_ALEXA_SMART_HOME

# === Internal dependency: homeassistant.components.automation ===
EVENT_AUTOMATION_TRIGGERED: str

# === Internal dependency: homeassistant.components.logbook ===
def log_entry(hass: HomeAssistant, name: str, message: str, domain: str | None = ..., entity_id: str | None = ..., context: Context | None = ...) -> None: ...
def async_log_entry(hass: HomeAssistant, name: str, message: str, domain: str | None = ..., entity_id: str | None = ..., context: Context | None = ...) -> None: ...
def _process_logbook_platform(hass: HomeAssistant, domain: str, platform: Any) -> None: ...
# re-export: from homeassistant.const import ATTR_DOMAIN
# re-export: from homeassistant.const import ATTR_ENTITY_ID
# re-export: from homeassistant.const import ATTR_NAME
# re-export: from homeassistant.const import EVENT_LOGBOOK_ENTRY
# re-export: from .const import ATTR_MESSAGE
# re-export: from .const import DOMAIN
CONFIG_SCHEMA: Schema

# === Internal dependency: homeassistant.components.logbook.models ===
class LazyEventPartialState: ...

# === Internal dependency: homeassistant.components.logbook.processor ===
class EventProcessor:
    def __init__(self, hass: HomeAssistant, event_types: tuple[EventType[Any] | str, ...], entity_ids: list[str] | None = ..., device_ids: list[str] | None = ..., context_id: str | None = ..., timestamp: bool = ..., include_entity_name: bool = ...) -> None: ...
    def get_events(self, start_day: dt, end_day: dt) -> list[dict[str, Any]]: ...

# === Internal dependency: homeassistant.components.logbook.queries.common ===
PSEUDO_EVENT_STATE_CHANGED: Final

# === Internal dependency: homeassistant.components.recorder ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.script ===
# re-export: from .const import EVENT_SCRIPT_STARTED

# === Internal dependency: homeassistant.components.sensor ===
# re-export: from .const import SensorStateClass

# === Internal dependency: homeassistant.const ===
CONF_DOMAINS: Final
CONF_ENTITIES: Final
CONF_EXCLUDE: Final
CONF_INCLUDE: Final
EVENT_CALL_SERVICE: Final
EVENT_HOMEASSISTANT_START: EventType[NoEventData]
EVENT_HOMEASSISTANT_STARTED: EventType[NoEventData]
EVENT_HOMEASSISTANT_STOP: EventType[NoEventData]
EVENT_LOGBOOK_ENTRY: Final
STATE_ON: Final
STATE_OFF: Final
ATTR_DOMAIN: Final
ATTR_SERVICE: Final
ATTR_NAME: Final
ATTR_ENTITY_ID: Final
ATTR_FRIENDLY_NAME: Final
ATTR_UNIT_OF_MEASUREMENT: Final

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id: str) -> tuple[str, str]: ...
def callback(func: _CallableT) -> _CallableT: ...
class HomeAssistant: ...
class Context:
    def __init__(self, user_id: str | None = ..., parent_id: str | None = ..., id: str | None = ...) -> None: ...
class State: ...
DOMAIN: str

# === Internal dependency: homeassistant.helpers.device_registry ===
CONNECTION_NETWORK_MAC: str

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass: HomeAssistant) -> EntityRegistry: ...

# === Internal dependency: homeassistant.helpers.entityfilter ===
CONF_ENTITY_GLOBS: str

# === Internal dependency: homeassistant.helpers.json ===
class JSONEncoder(json.JSONEncoder): ...

# === Internal dependency: homeassistant.setup ===
async def async_setup_component(hass: core.HomeAssistant, domain: str, config: ConfigType) -> bool: ...

# === Internal dependency: homeassistant.util.dt ===
def utc_to_timestamp(utc_dt: dt.datetime) -> float: ...
UTC: Any
utcnow: partial

# === Third-party dependency: pytest ===
# Used symbols: fixture, raises

# === Internal dependency: tests.common ===
class MockConfigEntry(config_entries.ConfigEntry):
    def __init__(self, *, data = ..., disabled_by = ..., domain = ..., entry_id = ..., minor_version = ..., options = ..., pref_disable_new_entities = ..., pref_disable_polling = ..., reason = ..., source = ..., state = ..., title = ..., unique_id = ..., version = ...) -> None: ...
    def add_to_hass(self, hass: HomeAssistant) -> None: ...
def mock_platform(hass: HomeAssistant, platform_path: str, module: Mock | MockPlatform | None = ..., built_in = ...) -> None: ...
def async_capture_events(hass: HomeAssistant, event_name: str) -> list[Event]: ...

# === Internal dependency: tests.components.logbook.common ===
class MockRow: ...
def mock_humanify(hass_, rows) -> Any: ...

# === Internal dependency: tests.components.recorder.common ===
async def async_wait_recording_done(hass: HomeAssistant) -> None: ...
async def async_recorder_block_till_done(hass: HomeAssistant) -> None: ...

# === Third-party dependency: voluptuous ===
# Used symbols: Invalid, Schema