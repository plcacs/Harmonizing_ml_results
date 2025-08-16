def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

def valid_stations(stations: list, given_stations: list) -> bool:
    ...

class NSDepartureSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by NS'
    _attr_icon: str = 'mdi:train'

    def __init__(self, nsapi: ns_api.NSAPI, name: str, departure: str, heading: str, via: str, time: datetime.time) -> None:
        ...

    @property
    def name(self) -> str:
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
