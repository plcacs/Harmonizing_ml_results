def _smart_type_name(_type: str) -> str:
    """Return a lowercase name of smart type."""
    if _type and _type == 'feelsLike':
        return 'feelslike'
    return _type

def _add_remove_devices() -> None:
    """Handle additions of devices and sensors."""

async def async_setup_entry(hass: HomeAssistant, entry: SensiboConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class SensiboMotionSensor(SensiboMotionBaseEntity, SensorEntity):
    def __init__(self, coordinator: SensiboDataUpdateCoordinator, device_id: str, sensor_id: str, sensor_data: Any, entity_description: SensiboMotionSensorEntityDescription) -> None:
    @property
    def native_value(self) -> StateType:

class SensiboDeviceSensor(SensiboDeviceBaseEntity, SensorEntity):
    def __init__(self, coordinator: SensiboDataUpdateCoordinator, device_id: str, entity_description: SensiboDeviceSensorEntityDescription) -> None:
    @property
    def native_value(self) -> StateType:
    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
