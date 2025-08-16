def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class RaspyRFMSwitch(SwitchEntity):
    _attr_should_poll: bool = False

    def __init__(self, raspyrfm_client: RaspyRFMClient, name: str, gateway: Gateway, controlunit: ControlUnit) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def assumed_state(self) -> bool:
        ...

    @property
    def is_on(self) -> bool:
        ...

    def turn_on(self, **kwargs) -> None:
        ...

    def turn_off(self, **kwargs) -> None:
        ...
