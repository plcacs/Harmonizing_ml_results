from typing import List, Union, Callable

def get_security_zone_device_class(zone: TotalConnectZone) -> Union[BinarySensorDeviceClass, None]:
async def async_setup_entry(hass: HomeAssistant, entry: TotalConnectConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
class TotalConnectZoneBinarySensor(TotalConnectZoneEntity, BinarySensorEntity):
    def __init__(self, coordinator: TotalConnectDataUpdateCoordinator, entity_description: TotalConnectZoneBinarySensorEntityDescription, zone: TotalConnectZone, location_id: str) -> None:
class TotalConnectAlarmBinarySensor(TotalConnectLocationEntity, BinarySensorEntity):
    def __init__(self, coordinator: TotalConnectDataUpdateCoordinator, entity_description: TotalConnectAlarmBinarySensorEntityDescription, location: TotalConnectLocation) -> None:
