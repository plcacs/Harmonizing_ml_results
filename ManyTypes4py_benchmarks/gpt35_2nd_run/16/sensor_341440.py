def get_next_departure(schedule: pygtfs.Schedule, start_station_id: str, end_station_id: str, offset: datetime.timedelta, include_tomorrow: bool = False) -> dict:
    ...

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class GTFSDepartureSensor(SensorEntity):
    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TIMESTAMP

    def __init__(self, gtfs: pygtfs.Schedule, name: str, origin: str, destination: str, offset: datetime.timedelta, include_tomorrow: bool) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    @property
    def icon(self) -> str:
        ...

    def update(self) -> None:
        ...

    def update_attributes(self) -> None:
        ...

    @staticmethod
    def dict_for_table(resource: Any) -> dict:
        ...

    def append_keys(self, resource: dict, prefix: str = None) -> None:
        ...

    def remove_keys(self, prefix: str) -> None:
        ...
