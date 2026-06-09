from typing import Any

# === Internal dependency: homeassistant.components.tag.const ===
DEVICE_ID = 'device_id'
DOMAIN = 'tag'
EVENT_TAG_SCANNED = 'tag_scanned'
TAG_ID = 'tag_id'
DEFAULT_NAME = 'Tag'
LOGGER = logging.getLogger(...)

# === Internal dependency: homeassistant.const ===
CONF_ID = 'id'
CONF_NAME = 'name'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

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
CHANGE_ADDED = 'added'
CHANGE_UPDATED = 'updated'
CHANGE_REMOVED = 'removed'

# === Internal dependency: homeassistant.helpers.config_validation ===
def string(value): ...
def datetime(value): ...
def empty_config_schema(domain): ...

# === Internal dependency: homeassistant.helpers.entity ===
class Entity:
    def suggested_object_id(self): ...
    def enabled(self): ...

# === Internal dependency: homeassistant.helpers.entity_component ===
class EntityComponent(Generic[_EntityT]): ...

# === Internal dependency: homeassistant.helpers.entity_registry ===
class RegistryEntry: ...
def async_get(hass): ...

# === Internal dependency: homeassistant.helpers.storage ===
class Store: ...

# === Internal dependency: homeassistant.helpers.typing ===
VolDictType: Any

# === Internal dependency: homeassistant.util ===
def slugify(text, *, separator=...): ...

# === Internal dependency: homeassistant.util.dt ===
def parse_datetime(dt_str): ...
def parse_datetime(dt_str, *, raise_on_error): ...
def parse_datetime(dt_str, *, raise_on_error=...): ...
UTC = dt.UTC
utcnow = partial(...)

# === Internal dependency: homeassistant.util.hass_dict ===
class HassKey(_Key[_T]):
    ...

# === Third-party dependency: voluptuous ===
# Used symbols: All, Length, Optional, Schema