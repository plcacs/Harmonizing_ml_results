def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class GitterSensor(SensorEntity):
    def __init__(self, data: GitterClient, room: str, name: str, username: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> int:
        ...

    @property
    def native_unit_of_measurement(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    def update(self) -> None:
        ...
