def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class VultrSensor(SensorEntity):
    def __init__(self, vultr: Vultr, subscription: str, name: str, description: SensorEntityDescription) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> float:
        ...

    def update(self) -> None:
        ...
