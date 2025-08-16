def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class IrishRailTransportSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by Irish Rail'
    _attr_icon: str = 'mdi:train'

    def __init__(self, data: IrishRailTransportData, station: str, direction: str, destination: str, stops_at: str, name: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        ...

    @property
    def native_unit_of_measurement(self) -> UnitOfTime:
        ...

    def update(self) -> None:
        ...

class IrishRailTransportData:
    def __init__(self, irish_rail: IrishRailRTPI, station: str, direction: str, destination: str, stops_at: str) -> None:
        ...

    def update(self) -> None:
        ...

    def _empty_train_data(self) -> list[dict[str, Any]]:
        ...
