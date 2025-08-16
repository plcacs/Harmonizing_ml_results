from typing import List, Optional
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

class MyUplinkDevicePointBinarySensor(MyUplinkEntity, BinarySensorEntity):
    def __init__(self, coordinator: MyUplinkDataCoordinator, device_id: str, device_point: DevicePoint, entity_description: Optional[BinarySensorEntityDescription], unique_id_suffix: str):
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def available(self) -> bool:
        ...

class MyUplinkDeviceBinarySensor(MyUplinkEntity, BinarySensorEntity):
    def __init__(self, coordinator: MyUplinkDataCoordinator, device_id: str, entity_description: BinarySensorEntityDescription, unique_id_suffix: str):
        ...

    @property
    def is_on(self) -> bool:
        ...

class MyUplinkSystemBinarySensor(MyUplinkSystemEntity, BinarySensorEntity):
    def __init__(self, coordinator: MyUplinkDataCoordinator, system_id: str, device_id: str, entity_description: BinarySensorEntityDescription, unique_id_suffix: str):
        ...

    @property
    def is_on(self) -> Optional[bool]:
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: MyUplinkConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
    ...
