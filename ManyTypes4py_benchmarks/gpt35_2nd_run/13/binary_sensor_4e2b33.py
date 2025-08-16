def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

def get_opening_type(zone: dict) -> str:
    ...

class Concord232ZoneSensor(BinarySensorEntity):
    def __init__(self, hass: HomeAssistant, client: concord232_client.Client, zone: dict, zone_type: str) -> None:
        ...

    @property
    def device_class(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def is_on(self) -> bool:
        ...

    def update(self) -> None:
        ...
