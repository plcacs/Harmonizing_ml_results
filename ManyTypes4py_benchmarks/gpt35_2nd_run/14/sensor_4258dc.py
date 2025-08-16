from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

def _sensor_unique_id(server_id: str, instance_num: int, suffix: str) -> str:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HyperionSensor(SensorEntity):
    def __init__(self, server_id: str, instance_num: int, instance_name: str, hyperion_client: Any, entity_description: SensorEntityDescription) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

class HyperionVisiblePrioritySensor(HyperionSensor):
    def __init__(self, server_id: str, instance_num: int, instance_name: str, hyperion_client: Any, entity_description: SensorEntityDescription) -> None:
        ...

    @callback
    def _update_priorities(self, _=None) -> None:
        ...
