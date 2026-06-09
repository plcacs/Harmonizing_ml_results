from typing import Any

# === Internal dependency: homeassistant.components.tag.const ===
DEVICE_ID: str
DOMAIN: str
EVENT_TAG_SCANNED: str
TAG_ID: str
DEFAULT_NAME: str
LOGGER: getLogger

# === Internal dependency: homeassistant.const ===
CONF_ID: Final
CONF_NAME: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception):
    ...

# === Internal dependency: homeassistant.helpers.collection ===
class ItemNotFound(CollectionError): ...
class IDManager:
    ...
class SerializedStorageCollection(TypedDict): ...
class DictStorageCollection(StorageCollection[dict, SerializedStorageCollection]):
class StorageCollectionWebsocket: ...
CHANGE_ADDED: str
CHANGE_UPDATED: str
CHANGE_REMOVED: str

# === Internal dependency: homeassistant.helpers.config_validation ===
def string(value: Any) -> str: ...
def datetime(value: Any) -> datetime_sys: ...
def empty_config_schema(domain: str) -> Callable[[dict], dict]: ...

# === Internal dependency: homeassistant.helpers.entity ===
class Entity:
    def suggested_object_id(self) -> str | None: ...
    def enabled(self) -> bool: ...

# === Internal dependency: homeassistant.helpers.entity_component ===
class EntityComponent(Generic[_EntityT]): ...

# === Internal dependency: homeassistant.helpers.entity_registry ===
class RegistryEntry: ...
def async_get(hass: HomeAssistant) -> EntityRegistry: ...

# === Internal dependency: homeassistant.helpers.storage ===
class Store: ...

# === Internal dependency: homeassistant.helpers.typing ===
VolDictType: Any

# === Internal dependency: homeassistant.util ===
def slugify(text: str | None, *, separator: str = ...) -> str: ...

# === Internal dependency: homeassistant.util.dt ===
def parse_datetime(dt_str: str) -> dt.datetime | None: ...
def parse_datetime(dt_str: str, *, raise_on_error: Literal[True]) -> dt.datetime: ...
def parse_datetime(dt_str: str, *, raise_on_error: Literal[False]) -> dt.datetime | None: ...
def parse_datetime(dt_str: str, *, raise_on_error: bool = ...) -> dt.datetime | None: ...
utcnow: partial

# === Internal dependency: homeassistant.util.hass_dict ===
class HassKey(_Key[_T]):
    ...

# === Third-party dependency: voluptuous ===
# Used symbols: All, Length, Optional, Schema