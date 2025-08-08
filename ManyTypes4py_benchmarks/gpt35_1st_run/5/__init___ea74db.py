from __future__ import annotations
from typing import List, Tuple, Any, Dict, Union
import datetime
import json
import logging
import growattServer
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, CONF_PASSWORD, CONF_URL, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import Throttle, dt as dt_util
from ..const import CONF_PLANT_ID, DEFAULT_PLANT_ID, DEFAULT_URL, DEPRECATED_URLS, DOMAIN, LOGIN_INVALID_AUTH_CODE
from .inverter import INVERTER_SENSOR_TYPES
from .mix import MIX_SENSOR_TYPES
from .sensor_entity_description import GrowattSensorEntityDescription
from .storage import STORAGE_SENSOR_TYPES
from .tlx import TLX_SENSOR_TYPES
from .total import TOTAL_SENSOR_TYPES

_LOGGER: logging.Logger
SCAN_INTERVAL: datetime.timedelta

def get_device_list(api: growattServer.GrowattApi, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class GrowattInverter(SensorEntity):
    _attr_has_entity_name: bool

    def __init__(self, probe: GrowattData, name: str, unique_id: str, description: GrowattSensorEntityDescription) -> None:
        ...

    @property
    def native_value(self) -> Union[float, int]:
        ...

    @property
    def native_unit_of_measurement(self) -> str:
        ...

    def update(self) -> None:
        ...

class GrowattData:
    def __init__(self, api: growattServer.GrowattApi, username: str, password: str, device_id: str, growatt_type: str) -> None:
        ...

    @Throttle(SCAN_INTERVAL)
    def update(self) -> None:
        ...

    def get_currency(self) -> str:
        ...

    def get_data(self, entity_description: GrowattSensorEntityDescription) -> Any:
        ...
