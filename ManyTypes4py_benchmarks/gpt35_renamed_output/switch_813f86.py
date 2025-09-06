from __future__ import annotations
from typing import Any, List, Dict, Optional
import voluptuous as vol
from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, ATTR_FRIENDLY_NAME, CONF_DEVICE_ID, CONF_NAME, CONF_SWITCHES, CONF_UNIQUE_ID, CONF_VALUE_TEMPLATE, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.script import Script
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_VALID_STATES: List[str] = [STATE_ON, STATE_OFF, 'true', 'false']
SWITCH_SCHEMA: vol.Schema = vol.All(cv.deprecated(ATTR_ENTITY_ID), vol.Schema({vol.Optional(CONF_VALUE_TEMPLATE): cv.template, vol.Required(CONF_TURN_ON): cv.SCRIPT_SCHEMA, vol.Required(CONF_TURN_OFF): cv.SCRIPT_SCHEMA, vol.Optional(ATTR_FRIENDLY_NAME): cv.string, vol.Optional(ATTR_ENTITY_ID): cv.entity_ids, vol.Optional(CONF_UNIQUE_ID): cv.string}).extend(TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY.schema))
PLATFORM_SCHEMA: vol.Schema = SWITCH_PLATFORM_SCHEMA.extend({vol.Required(CONF_SWITCHES): cv.schema_with_slug_keys(SWITCH_SCHEMA)})
SWITCH_CONFIG_SCHEMA: vol.Schema = vol.Schema({vol.Required(CONF_NAME): cv.template, vol.Optional(CONF_VALUE_TEMPLATE): cv.template, vol.Optional(CONF_TURN_ON): cv.SCRIPT_SCHEMA, vol.Optional(CONF_TURN_OFF): cv.SCRIPT_SCHEMA, vol.Optional(CONF_DEVICE_ID): selector.DeviceSelector()})

async def func_9tifpxx2(hass: HomeAssistant, config: ConfigType) -> List[SwitchTemplate]:
    ...

async def func_ii2xrofa(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

async def func_eixt12i8(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    ...

@callback
def func_tzulejuf(hass: HomeAssistant, name: str, config: Dict[str, Any]) -> SwitchTemplate:
    ...

class SwitchTemplate(TemplateEntity, SwitchEntity, RestoreEntity):
    def __init__(self, hass: HomeAssistant, object_id: Optional[str], config: Dict[str, Any], unique_id: Optional[str]) -> None:
        ...

    async def func_im8dooj3(self) -> None:
        ...

    @callback
    def func_izcu8f5g(self) -> None:
        ...

    @property
    def func_1fdvq74o(self) -> bool:
        ...

    async def func_ppfd4cd3(self, **kwargs: Any) -> None:
        ...

    async def func_q0ayqqjg(self, **kwargs: Any) -> None:
        ...
