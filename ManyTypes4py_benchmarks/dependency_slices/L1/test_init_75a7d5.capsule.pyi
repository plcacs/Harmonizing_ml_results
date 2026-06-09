# === Third-party dependency: freezegun ===
# Used symbols: freeze_time

# === Internal dependency: homeassistant.components.alexa.smart_home ===
from .const import EVENT_ALEXA_SMART_HOME

# === Internal dependency: homeassistant.components.automation ===
EVENT_AUTOMATION_TRIGGERED = 'automation_triggered'

# === Internal dependency: homeassistant.components.logbook ===
def log_entry(hass, name, message, domain=..., entity_id=..., context=...): ...
def async_log_entry(hass, name, message, domain=..., entity_id=..., context=...): ...
def _process_logbook_platform(hass, domain, platform): ...
from homeassistant.const import ATTR_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.const import ATTR_NAME
from homeassistant.const import EVENT_LOGBOOK_ENTRY
from .const import ATTR_MESSAGE
from .const import DOMAIN
CONFIG_SCHEMA = vol.Schema(...)

# === Internal dependency: homeassistant.components.logbook.models ===
class LazyEventPartialState: ...

# === Internal dependency: homeassistant.components.logbook.processor ===
class EventProcessor:
    def __init__(self, hass, event_types, entity_ids=..., device_ids=..., context_id=..., timestamp=..., include_entity_name=...): ...
    def get_events(self, start_day, end_day): ...

# === Internal dependency: homeassistant.components.logbook.queries.common ===
PSEUDO_EVENT_STATE_CHANGED = None

# === Internal dependency: homeassistant.components.recorder ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.script ===
from .const import EVENT_SCRIPT_STARTED

# === Internal dependency: homeassistant.components.sensor ===
from .const import SensorStateClass

# === Internal dependency: homeassistant.const ===
CONF_DOMAINS = 'domains'
CONF_ENTITIES = 'entities'
CONF_EXCLUDE = 'exclude'
CONF_INCLUDE = 'include'
EVENT_CALL_SERVICE = 'call_service'
EVENT_HOMEASSISTANT_START = EventType(...)
EVENT_HOMEASSISTANT_STARTED = EventType(...)
EVENT_HOMEASSISTANT_STOP = EventType(...)
EVENT_LOGBOOK_ENTRY = 'logbook_entry'
STATE_ON = 'on'
STATE_OFF = 'off'
ATTR_DOMAIN = 'domain'
ATTR_SERVICE = 'service'
ATTR_NAME = 'name'
ATTR_ENTITY_ID = 'entity_id'
ATTR_FRIENDLY_NAME = 'friendly_name'
ATTR_UNIT_OF_MEASUREMENT = 'unit_of_measurement'

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id): ...
def callback(func): ...
class HomeAssistant: ...
class Context:
    def __init__(self, user_id=..., parent_id=..., id=...): ...
class State: ...
DOMAIN = 'homeassistant'

# === Internal dependency: homeassistant.helpers.device_registry ===
CONNECTION_NETWORK_MAC = 'mac'

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass): ...

# === Internal dependency: homeassistant.helpers.entityfilter ===
CONF_ENTITY_GLOBS = 'entity_globs'

# === Internal dependency: homeassistant.helpers.json ===
class JSONEncoder(json.JSONEncoder): ...

# === Internal dependency: homeassistant.setup ===
async def async_setup_component(hass, domain, config): ...

# === Internal dependency: homeassistant.util.dt ===
def utc_to_timestamp(utc_dt): ...
UTC = dt.UTC
utcnow = partial(...)

# === Internal dependency: homeassistant.util.event_type ===
class EventType(Generic[_DataT]): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, raises

# === Internal dependency: tests.common ===
class MockConfigEntry(config_entries.ConfigEntry):
    def __init__(self, *, data=..., disabled_by=..., domain=..., entry_id=..., minor_version=..., options=..., pref_disable_new_entities=..., pref_disable_polling=..., reason=..., source=..., state=..., title=..., unique_id=..., version=...): ...
    def add_to_hass(self, hass): ...
def mock_platform(hass, platform_path, module=..., built_in=...): ...
def async_capture_events(hass, event_name): ...

# === Internal dependency: tests.components.logbook.common ===
class MockRow: ...
def mock_humanify(hass_, rows): ...

# === Internal dependency: tests.components.recorder.common ===
async def async_wait_recording_done(hass): ...
async def async_recorder_block_till_done(hass): ...

# === Third-party dependency: voluptuous ===
# Used symbols: Invalid, Schema