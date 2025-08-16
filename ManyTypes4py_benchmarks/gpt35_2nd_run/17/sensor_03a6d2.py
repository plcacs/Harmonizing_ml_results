def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class NOAATidesData(TypedDict):
    time_stamp: List[Timestamp]
    hi_lo: List[str]
    predicted_wl: List[float]

class NOAATidesAndCurrentsSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by NOAA'

    def __init__(self, name: str, station_id: str, timezone: str, unit_system: str, station: coops.Station) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    @property
    def native_value(self) -> Optional[str]:
        ...

    def update(self) -> None:
        ...
