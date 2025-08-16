def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class WorldTidesInfoSensor(SensorEntity):
    _attr_attribution: str = ATTRIBUTION

    def __init__(self, name: str, lat: float, lon: float, key: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    @property
    def native_value(self) -> str:
        ...

    def update(self) -> None:
        ...
