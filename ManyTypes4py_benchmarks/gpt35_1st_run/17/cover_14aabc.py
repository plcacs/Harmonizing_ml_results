def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
async def _async_create_entities(hass: HomeAssistant, config: ConfigType) -> List[CoverTemplate]:
class CoverTemplate(TemplateEntity, CoverEntity):
    def __init__(self, hass: HomeAssistant, object_id: str, config: ConfigType, unique_id: str) -> None:
    def _async_setup_templates(self) -> None:
    def _update_state(self, result: Any) -> None:
    def _update_position(self, result: Any) -> None:
    def _update_tilt(self, result: Any) -> None:
    @property
    def is_closed(self) -> Optional[bool]:
    @property
    def is_opening(self) -> bool:
    @property
    def is_closing(self) -> bool:
    @property
    def current_cover_position(self) -> Optional[int]:
    @property
    def current_cover_tilt_position(self) -> Optional[int]:
    async def async_open_cover(self, **kwargs: Any) -> None:
    async def async_close_cover(self, **kwargs: Any) -> None:
    async def async_stop_cover(self, **kwargs: Any) -> None:
    async def async_set_cover_position(self, **kwargs: Any) -> None:
    async def async_open_cover_tilt(self, **kwargs: Any) -> None:
    async def async_close_cover_tilt(self, **kwargs: Any) -> None:
    async def async_set_cover_tilt_position(self, **kwargs: Any) -> None:
