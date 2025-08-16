from __future__ import annotations
from typing import List, Optional
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.core import HomeAssistant
from . import DevoloHomeControlConfigEntry

async def async_setup_entry(hass: HomeAssistant, entry: DevoloHomeControlConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    entities: List[DevoloMultiLevelDeviceEntity] = []
    ...

class DevoloMultiLevelDeviceEntity(DevoloDeviceEntity, SensorEntity):
    @property
    def native_value(self) -> Optional[float]:
        ...

class DevoloGenericMultiLevelDeviceEntity(DevoloMultiLevelDeviceEntity):
    def __init__(self, homecontrol: HomeControl, device_instance: Zwave, element_uid: str) -> None:
        ...

class DevoloBatteryEntity(DevoloMultiLevelDeviceEntity):
    def __init__(self, homecontrol: HomeControl, device_instance: Zwave, element_uid: str) -> None:
        ...

class DevoloConsumptionEntity(DevoloMultiLevelDeviceEntity):
    def __init__(self, homecontrol: HomeControl, device_instance: Zwave, element_uid: str, consumption: str) -> None:
        ...

    @property
    def unique_id(self) -> str:
        ...

    def _sync(self, message: List[str]) -> None:
        ...
