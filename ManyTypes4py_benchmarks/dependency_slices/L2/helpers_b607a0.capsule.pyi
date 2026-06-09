from typing import Any

# === Internal dependency: homeassistant.components.sensor ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.zwave_js.const ===
DOMAIN: str
DATA_CLIENT: str
DATA_OLD_SERVER_LOG_LEVEL: str
LOGGER: getLogger
LIB_LOGGER: getLogger
ATTR_ENDPOINT: str
ATTR_COMMAND_CLASS: str
ATTR_PROPERTY: str
ATTR_PROPERTY_KEY: str

# === Internal dependency: homeassistant.config_entries ===
class ConfigEntryState(Enum): ...
class ConfigEntry(Generic[_DataT]): ...

# === Internal dependency: homeassistant.const ===
__version__: Final
CONF_TYPE: Final
ATTR_ENTITY_ID: Final
ATTR_AREA_ID: Final
ATTR_DEVICE_ID: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.device_registry ===
class DeviceInfo(TypedDict): ...
def async_get(hass: HomeAssistant) -> DeviceRegistry: ...
def async_entries_for_area(registry: DeviceRegistry, area_id: str) -> list[DeviceEntry]: ...

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass: HomeAssistant) -> EntityRegistry: ...
def async_entries_for_area(registry: EntityRegistry, area_id: str) -> list[RegistryEntry]: ...

# === Internal dependency: homeassistant.helpers.group ===
def expand_entity_ids(hass: HomeAssistant, entity_ids: Iterable[Any]) -> list[str]: ...

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