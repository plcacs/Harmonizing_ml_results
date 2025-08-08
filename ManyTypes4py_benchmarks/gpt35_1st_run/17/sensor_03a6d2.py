def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class NOAATidesData(TypedDict):
    time_stamp: List[Timestamp]
    hi_lo: List[str]
    predicted_wl: List[float]

class NOAATidesAndCurrentsSensor(SensorEntity):
    _attr_attribution: str = 'Data provided by NOAA'
    _name: str
    _station_id: str
    _timezone: str
    _unit_system: str
    _station: coops.Station
    data: Optional[NOAATidesData]
    _attr_unique_id: str

    def __init__(self, name: str, station_id: str, timezone: str, unit_system: str, station: coops.Station) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        ...

    @property
    def native_value(self) -> Optional[str]:
        ...

    def update(self) -> None:
        ...
