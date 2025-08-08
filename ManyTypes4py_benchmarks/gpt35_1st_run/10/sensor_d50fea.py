def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class NeurioData:
    def __init__(self, api_key: str, api_secret: str, sensor_id: str) -> None:
        ...

    @property
    def daily_usage(self) -> float:
        ...

    @property
    def active_power(self) -> float:
        ...

    def get_active_power(self) -> None:
        ...

    def get_daily_usage(self) -> None:
        ...

class NeurioEnergy(SensorEntity):
    def __init__(self, data: NeurioData, name: str, sensor_type: str, update_call: Callable[[], None]) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> float:
        ...

    @property
    def native_unit_of_measurement(self) -> str:
        ...

    def update(self) -> None:
        ...
