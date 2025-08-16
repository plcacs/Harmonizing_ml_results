from typing import List, Optional, Tuple

def device_class_and_uom(data: object, entity_description: DSMRSensorEntityDescription) -> Tuple[SensorDeviceClass, Optional[UnitOfEnergy]]:
    ...

def rename_old_gas_to_mbus(hass: HomeAssistant, entry: ConfigEntry, mbus_device_id: str) -> None:
    ...

def is_supported_description(data: object, description: DSMRSensorEntityDescription, dsmr_version: str) -> bool:
    ...

def create_mbus_entities(hass: HomeAssistant, telegram: Telegram, entry: ConfigEntry, dsmr_version: str) -> Generator[DSMREntity, None, None]:
    ...

def get_dsmr_object(telegram: Telegram, mbus_id: int, obis_reference: str) -> Optional[DSMRObject]:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class DSMREntity(SensorEntity):
    def __init__(self, entity_description: DSMRSensorEntityDescription, entry: ConfigEntry, telegram: Telegram, device_class: SensorDeviceClass, native_unit_of_measurement: Optional[UnitOfVolume], serial_id: str = '', mbus_id: int = 0) -> None:
        ...

    def update_data(self, telegram: Telegram) -> None:
        ...

    def get_dsmr_object_attr(self, attribute: str) -> Optional[str]:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def native_value(self) -> Optional[float]:
        ...

    @staticmethod
    def translate_tariff(value: str, dsmr_version: str) -> Optional[str]:
        ...
