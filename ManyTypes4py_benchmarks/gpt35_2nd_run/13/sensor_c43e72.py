from __future__ import annotations
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union
from sml import SmlGetListResponse
from sml.asyncio import SmlProtocol
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import DEGREE, UnitOfElectricCurrent, UnitOfElectricPotential, UnitOfEnergy, UnitOfFrequency, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.dt import utcnow
from .const import CONF_SERIAL_PORT, DEFAULT_DEVICE_NAME, DOMAIN, LOGGER, SIGNAL_EDL21_TELEGRAM

MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(seconds=60)
SENSOR_TYPES: List[SensorEntityDescription] = [SensorEntityDescription(key='1-0:0.0.0*255', translation_key='ownership_id', entity_registry_enabled_default=False), ...]

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class EDL21:
    _OBIS_BLACKLIST: set[str] = {'1-0:96.50.1*1', '1-0:96.50.1*4', ...}

    def __init__(self, hass: HomeAssistant, config: Dict[str, Any], async_add_entities: AddConfigEntryEntitiesCallback) -> None:
        ...

    async def connect(self) -> None:
        ...

    def event(self, message_body: SmlGetListResponse) -> None:
        ...

class EDL21Entity(SensorEntity):
    _attr_should_poll: bool = False
    _attr_has_entity_name: bool = True

    def __init__(self, electricity_id: str, obis: str, entity_description: SensorEntityDescription, telegram: Dict[str, Any]) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def native_unit_of_measurement(self) -> Optional[Union[UnitOfEnergy, UnitOfPower, UnitOfElectricCurrent, UnitOfElectricPotential, DEGREE, UnitOfFrequency]]:
        ...
