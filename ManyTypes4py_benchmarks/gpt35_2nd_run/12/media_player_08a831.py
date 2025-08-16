def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

def service_handle(service: ServiceCall) -> None:
    ...

class BlackbirdZone(MediaPlayerEntity):
    _attr_supported_features: int = MediaPlayerEntityFeature.TURN_ON | MediaPlayerEntityFeature.TURN_OFF | MediaPlayerEntityFeature.SELECT_SOURCE

    def __init__(self, blackbird, sources: dict[int, str], zone_id: int, zone_name: str) -> None:
        ...

    def update(self) -> None:
        ...

    @property
    def media_title(self) -> str:
        ...

    def set_all_zones(self, source: str) -> None:
        ...

    def select_source(self, source: str) -> None:
        ...

    def turn_on(self) -> None:
        ...

    def turn_off(self) -> None:
        ...
