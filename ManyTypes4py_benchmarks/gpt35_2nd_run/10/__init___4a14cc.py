from homeassistant.helpers.entity import ToggleEntity
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.script import Script
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import bind_hass
from homeassistant.util.async_ import create_eager_task
from homeassistant.util.dt import parse_datetime
from typing import Any

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

@bind_hass
def is_on(hass: HomeAssistant, entity_id: str) -> bool:
    ...

async def _async_process_config(hass: HomeAssistant, config: ConfigType, component: EntityComponent) -> None:
    ...

async def _prepare_script_config(hass: HomeAssistant, config: ConfigType) -> list:
    ...

async def _create_script_entities(hass: HomeAssistant, script_configs: list) -> list:
    ...

class BaseScriptEntity(ToggleEntity, ABC):
    ...

class UnavailableScriptEntity(BaseScriptEntity):
    ...

class ScriptEntity(BaseScriptEntity, RestoreEntity):
    ...

@websocket_api.websocket_command({'type': 'script/config', 'entity_id': str})
def websocket_config(hass: HomeAssistant, connection: Any, msg: dict) -> None:
    ...
