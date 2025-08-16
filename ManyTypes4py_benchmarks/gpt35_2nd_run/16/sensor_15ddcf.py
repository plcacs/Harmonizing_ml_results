def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class TMBSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by Transport Metropolitans de Barcelona'
    _attr_icon: str = 'mdi:bus-clock'

    def __init__(self, ibus_client: IBus, stop: str, line: str, name: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_unit_of_measurement(self) -> UnitOfTime:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def native_value(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        ...
