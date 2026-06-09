from typing import Any

# === Internal dependency: homeassistant.components.group.const ===
DOMAIN: str
REG_KEY: Any
GROUP_ORDER: str
ATTR_AUTO: str
ATTR_ORDER: str

# === Internal dependency: homeassistant.components.group.registry ===
class SingleStateType: ...
class GroupIntegrationRegistry: ...

# === Internal dependency: homeassistant.const ===
STATE_ON: Final
STATE_OFF: Final
ATTR_ENTITY_ID: Final
ATTR_ASSUMED_STATE: Final

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id: str) -> tuple[str, str]: ...
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.helpers.entity ===
def async_generate_entity_id(entity_id_format: str, name: str | None, current_ids: Iterable[str] | None = ..., hass: HomeAssistant | None = ...) -> str: ...
class Entity:
    def suggested_object_id(self) -> str | None: ...
    def enabled(self) -> bool: ...

# === Internal dependency: homeassistant.helpers.entity_component ===
class EntityComponent(Generic[_EntityT]): ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_state_change_event(hass: HomeAssistant, entity_ids: str | Iterable[str], action: Callable[[Event[EventStateChangedData]], Any], job_type: HassJobType | None = ...) -> CALLBACK_TYPE: ...

# === Internal dependency: homeassistant.helpers.start ===
def async_at_start(hass: HomeAssistant, at_start_cb: Callable[[HomeAssistant], Coroutine[Any, Any, None] | None]) -> CALLBACK_TYPE: ...