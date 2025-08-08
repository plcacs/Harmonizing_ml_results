def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class PALoopbackSwitch(SwitchEntity):
    def __init__(self, name: str, pa_server: Pulse, sink_name: str, source_name: str) -> None:
        ...

    def _get_module_idx(self) -> Any:
        ...

    @property
    def available(self) -> bool:
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
