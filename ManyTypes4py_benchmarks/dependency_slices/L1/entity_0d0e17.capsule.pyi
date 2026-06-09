# === Internal dependency: homeassistant.components.group.const ===
DOMAIN = 'group'
REG_KEY = f'{DOMAIN}_registry'
GROUP_ORDER = 'group_order'
ATTR_AUTO = 'auto'
ATTR_ORDER = 'order'

# === Internal dependency: homeassistant.components.group.registry ===
class SingleStateType: ...
class GroupIntegrationRegistry: ...

# === Internal dependency: homeassistant.const ===
STATE_ON = 'on'
STATE_OFF = 'off'
ATTR_ENTITY_ID = 'entity_id'
ATTR_ASSUMED_STATE = 'assumed_state'

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id): ...
def callback(func): ...

# === Internal dependency: homeassistant.helpers.entity ===
def async_generate_entity_id(entity_id_format, name, current_ids=..., hass=...): ...
class Entity:
    def suggested_object_id(self): ...
    def enabled(self): ...

# === Internal dependency: homeassistant.helpers.entity_component ===
class EntityComponent(Generic[_EntityT]): ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_state_change_event(hass, entity_ids, action, job_type=...): ...

# === Internal dependency: homeassistant.helpers.start ===
def async_at_start(hass, at_start_cb): ...