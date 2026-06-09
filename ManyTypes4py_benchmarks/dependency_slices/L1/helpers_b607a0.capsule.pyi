from typing import Any

# === Internal dependency: homeassistant.components.sensor ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.zwave_js.const ===
DOMAIN = 'zwave_js'
DATA_CLIENT = 'client'
DATA_OLD_SERVER_LOG_LEVEL = 'old_server_log_level'
LOGGER = logging.getLogger(...)
LIB_LOGGER = logging.getLogger(...)
ATTR_ENDPOINT = 'endpoint'
ATTR_COMMAND_CLASS = 'command_class'
ATTR_PROPERTY = 'property'
ATTR_PROPERTY_KEY = 'property_key'

# === Internal dependency: homeassistant.config_entries ===
class ConfigEntryState(Enum): ...
class ConfigEntry(Generic[_DataT]): ...

# === Internal dependency: homeassistant.const ===
MAJOR_VERSION = 2024
MINOR_VERSION = 8
PATCH_VERSION = '0.dev0'
__short_version__ = f'{MAJOR_VERSION}.{MINOR_VERSION}'
__version__ = f'{__short_version__}.{PATCH_VERSION}'
CONF_TYPE = 'type'
ATTR_ENTITY_ID = 'entity_id'
ATTR_AREA_ID = 'area_id'
ATTR_DEVICE_ID = 'device_id'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.device_registry ===
class DeviceInfo(TypedDict): ...
def async_get(hass): ...
def async_entries_for_area(registry, area_id): ...

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass): ...
def async_entries_for_area(registry, area_id): ...

# === Internal dependency: homeassistant.helpers.group ===
def expand_entity_ids(hass, entity_ids): ...

# === Internal dependency: homeassistant.helpers.typing ===
ConfigType: Any
VolSchemaType: Any

# === Third-party dependency: voluptuous ===
# Used symbols: All, Coerce, In, Invalid, Range

# === Unresolved dependency: zwave_js_server.client ===
# Used unresolved symbols: Client

# === Unresolved dependency: zwave_js_server.const ===
# Used unresolved symbols: CommandClass, ConfigurationValueType, LOG_LEVEL_MAP, LogLevel

# === Unresolved dependency: zwave_js_server.model.log_config ===
# Used unresolved symbols: LogConfig

# === Unresolved dependency: zwave_js_server.model.node ===
# Used unresolved symbols: Node

# === Unresolved dependency: zwave_js_server.model.value ===
# Used unresolved symbols: ConfigurationValue, Value, get_value_id_str