def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class PencomRelay(SwitchEntity):
    def __init__(self, hub: Pencompy, board: int, addr: int, name: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def is_on(self) -> bool:
        ...

    def turn_on(self, **kwargs: Any) -> None:
        ...

    def turn_off(self, **kwargs: Any) -> None:
        ...

    def update(self) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...
