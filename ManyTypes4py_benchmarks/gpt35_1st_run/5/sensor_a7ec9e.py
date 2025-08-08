def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class HpIloSensor(SensorEntity):
    def __init__(self, hass: HomeAssistant, hp_ilo_data: HpIloData, sensor_type: str, sensor_name: str, sensor_value_template: str, unit_of_measurement: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_unit_of_measurement(self) -> str:
        ...

    @property
    def native_value(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    def update(self) -> None:
        ...

class HpIloData:
    def __init__(self, host: str, port: int, login: str, password: str) -> None:
        ...

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        ...
