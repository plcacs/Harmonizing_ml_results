def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

def get_stream_source(guid: str, client: QVRClient) -> Optional[str]:
    ...

class QVRProCamera(Camera):
    def __init__(self, name: str, model: str, brand: str, channel_index: int, guid: str, stream_source: str, client: QVRClient) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def model(self) -> str:
        ...

    @property
    def brand(self) -> str:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, str]:
        ...

    def camera_image(self, width: Optional[int] = None, height: Optional[int] = None) -> bytes:
        ...

    async def stream_source(self) -> str:
        ...
