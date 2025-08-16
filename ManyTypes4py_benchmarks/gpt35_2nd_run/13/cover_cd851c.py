def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class SCSGateCover(CoverEntity):
    def __init__(self, scs_id: str, name: str, logger: logging.Logger, scsgate: Any) -> None:
        ...

    @property
    def scs_id(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    def open_cover(self, **kwargs: Any) -> None:
        ...

    def close_cover(self, **kwargs: Any) -> None:
        ...

    def stop_cover(self, **kwargs: Any) -> None:
        ...

    def process_event(self, message: Any) -> None:
        ...
