from typing import Any

# === Internal dependency: homeassistant.components.binary_sensor ===
class BinarySensorDeviceClass(StrEnum): ...

# === Internal dependency: homeassistant.components.homeassistant.const ===
DATA_EXPOSED_ENTITIES: HassKey[ExposedEntities]
DOMAIN: Any

# === Internal dependency: homeassistant.components.sensor ===
# re-export: from .const import SensorDeviceClass

# === Internal dependency: homeassistant.components.websocket_api ===
def async_register_command(hass: HomeAssistant, command_or_handler: str | const.WebSocketCommandHandler, handler: const.WebSocketCommandHandler | None = ..., schema: VolSchemaType | None = ...) -> None: ...
# re-export: from .const import ERR_NOT_ALLOWED
# re-export: from .decorators import require_admin
# re-export: from .decorators import websocket_command

# === Internal dependency: homeassistant.const ===
CLOUD_NEVER_EXPOSED_ENTITIES: Final[list[str]]

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id: str) -> tuple[str, str]: ...
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.entity ===
def get_device_class(hass: HomeAssistant, entity_id: str) -> str | None: ...

# === Internal dependency: homeassistant.helpers.entity_registry ===
def async_get(hass: HomeAssistant) -> EntityRegistry: ...

# === Internal dependency: homeassistant.helpers.storage ===
class Store:
    def __init__(self, hass: HomeAssistant, version: int, key: str, private: bool = ..., *, atomic_writes: bool = ..., encoder: type[JSONEncoder] | None = ..., minor_version: int = ..., read_only: bool = ...) -> None: ...

# === Internal dependency: homeassistant.util.read_only_dict ===
class ReadOnlyDict(dict[_KT, _VT]): ...

# === Third-party dependency: voluptuous ===
# Used symbols: In, Required