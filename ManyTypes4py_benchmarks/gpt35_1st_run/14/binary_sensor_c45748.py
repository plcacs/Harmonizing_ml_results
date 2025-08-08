from typing import Any, Dict, List, Optional
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.restore_state import RestoreEntity

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class XiaomiBinarySensor(XiaomiDevice, BinarySensorEntity):
    def __init__(self, device: Dict[str, Any], name: str, xiaomi_hub: Any, data_key: str, device_class: Optional[str], config_entry: ConfigEntry) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def device_class(self) -> Optional[str]:
        ...

    def update(self) -> None:
        ...

class XiaomiNatgasSensor(XiaomiBinarySensor):
    def __init__(self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...

class XiaomiMotionSensor(XiaomiBinarySensor):
    def __init__(self, device: Dict[str, Any], hass: HomeAssistant, xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    @callback
    def _async_set_no_motion(self, now: Any) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...

class XiaomiDoorSensor(XiaomiBinarySensor, RestoreEntity):
    def __init__(self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...

class XiaomiWaterLeakSensor(XiaomiBinarySensor):
    def __init__(self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...

class XiaomiSmokeSensor(XiaomiBinarySensor):
    def __init__(self, device: Dict[str, Any], xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...

class XiaomiVibration(XiaomiBinarySensor):
    def __init__(self, device: Dict[str, Any], name: str, data_key: str, xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...

class XiaomiButton(XiaomiBinarySensor):
    def __init__(self, device: Dict[str, Any], name: str, data_key: str, hass: HomeAssistant, xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...

class XiaomiCube(XiaomiBinarySensor):
    def __init__(self, device: Dict[str, Any], hass: HomeAssistant, xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def parse_data(self, data: Dict[str, Any], raw_data: Dict[str, Any]) -> bool:
        ...
