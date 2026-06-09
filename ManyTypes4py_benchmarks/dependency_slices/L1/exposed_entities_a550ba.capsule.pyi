# === Internal dependency: homeassistant.components.binary_sensor ===
class BinarySensorDeviceClass(StrEnum): ...

# === Internal dependency: homeassistant.components.homeassistant.const ===
DOMAIN = ha.DOMAIN
DATA_EXPOSED_ENTITIES = HassKey(...)

# === Internal dependency: homeassistant.components.sensor ===
from .const import SensorDeviceClass

# === Internal dependency: homeassistant.components.websocket_api ===
def async_register_command(hass, command_or_handler, handler=..., schema=...): ...
from .const import ERR_NOT_ALLOWED
from .decorators import require_admin
from .decorators import websocket_command

# === Internal dependency: homeassistant.const ===
CLOUD_NEVER_EXPOSED_ENTITIES = ['group.all_locks']

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id): ...
def callback(func): ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.entity ===
def get_device_class(hass, entity_id): ...

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass): ...

# === Internal dependency: homeassistant.helpers.storage ===
class Store:
    def __init__(self, hass, version, key, private=..., *, atomic_writes=..., encoder=..., minor_version=..., read_only=...): ...

# === Internal dependency: homeassistant.util.hass_dict ===
class HassKey(_Key[_T]): ...

# === Internal dependency: homeassistant.util.read_only_dict ===
class ReadOnlyDict(dict[_KT, _VT]): ...

# === Third-party dependency: voluptuous ===
# Used symbols: In, Required