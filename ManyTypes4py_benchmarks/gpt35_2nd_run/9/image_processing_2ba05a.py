def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class Doods(ImageProcessingEntity):
    def __init__(self, hass: HomeAssistant, camera_entity: str, name: str, doods: PyDOODS, detector: dict, config: ConfigType) -> None:
        ...

    @property
    def camera_entity(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def state(self) -> int:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    def _save_image(self, image: bytes, matches: dict, paths: list[str]) -> None:
        ...

    def process_image(self, image: bytes) -> None:
        ...
