async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class TibberSensor(SensorEntity):
    def __init__(self, *args: Any, tibber_home: tibber.Home, **kwargs: Any) -> None:

    @property
    def device_info(self) -> DeviceInfo:

class TibberSensorElPrice(TibberSensor):
    def __init__(self, tibber_home: tibber.Home) -> None:

    async def async_update(self) -> None:

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    async def _fetch_data(self) -> None:

class TibberDataSensor(TibberSensor, CoordinatorEntity[TibberDataCoordinator]):
    def __init__(self, tibber_home: tibber.Home, coordinator: TibberDataCoordinator, entity_description: SensorEntityDescription) -> None:

    @property
    def native_value(self) -> StateType:

class TibberSensorRT(TibberSensor, CoordinatorEntity['TibberRtDataCoordinator']):
    def __init__(self, tibber_home: tibber.Home, description: SensorEntityDescription, initial_state: StateType, coordinator: TibberRtDataCoordinator) -> None:

    @property
    def available(self) -> bool:

    @callback
    def _handle_coordinator_update(self) -> None:

class TibberRtEntityCreator:
    def __init__(self, async_add_entities: AddConfigEntryEntitiesCallback, tibber_home: tibber.Home, entity_registry: er) -> None:

    @callback
    def _migrate_unique_id(self, sensor_description: SensorEntityDescription) -> None:

    @callback
    def add_sensors(self, coordinator: TibberRtDataCoordinator, live_measurement: dict) -> None:

class TibberRtDataCoordinator(DataUpdateCoordinator):
    def __init__(self, add_sensor_callback: Callable, tibber_home: tibber.Home, hass: HomeAssistant) -> None:

    @callback
    def _handle_ha_stop(self, _event: Event) -> None:

    @callback
    def _data_updated(self) -> None:

    def get_live_measurement(self) -> dict:
