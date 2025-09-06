from __future__ import annotations
import logging
from typing import Any, List, Optional, Dict
from homeassistant.components import mqtt
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.const import DEGREE, UnitOfPrecipitationDepth, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import slugify
from homeassistant.util.json import json_loads_object

_LOGGER: logging.Logger = logging.getLogger(__name__)
DOMAIN: str = 'arwn'
DATA_ARWN: str = 'arwn'
TOPIC: str = 'arwn/#'

def func_kdo2l6kl(topic: str, payload: Dict[str, Any]) -> List[ArwnSensor]:
    ...

def func_jxfuqghd(name: str) -> str:
    ...

async def func_bmsdrgv7(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class ArwnSensor(SensorEntity):
    _attr_should_poll: bool = False

    def __init__(self, topic: str, name: str, state_key: str, units: str, icon: Optional[str] = None, device_class: Optional[SensorDeviceClass] = None) -> None:
        ...

    def func_278d5096(self, event: Dict[str, Any]) -> None:
        ...
