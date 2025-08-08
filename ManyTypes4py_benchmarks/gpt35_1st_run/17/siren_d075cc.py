def supported(event: Any) -> bool:
    ...

def get_first_key(data: dict, entry: Any) -> Any:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class RfxtrxOffDelayMixin(Entity):
    _timeout: CALLBACK_TYPE | None = None
    _off_delay: int | None = None

    def _setup_timeout(self) -> None:
        ...

    def _cancel_timeout(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

class RfxtrxChime(RfxtrxCommandEntity, SirenEntity, RfxtrxOffDelayMixin):
    _attr_supported_features: int = SirenEntityFeature.TURN_ON | SirenEntityFeature.TONES

    def __init__(self, device: DeviceTuple, device_id: str, off_delay: int | None = None, event: Any = None) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    def _apply_event(self, event: Any) -> None:
        ...

    @callback
    def _handle_event(self, event: Any, device_id: str) -> None:
        ...

class RfxtrxSecurityPanic(RfxtrxCommandEntity, SirenEntity, RfxtrxOffDelayMixin):
    _attr_supported_features: int = SirenEntityFeature.TURN_ON | SirenEntityFeature.TURN_OFF

    def __init__(self, device: DeviceTuple, device_id: str, off_delay: int | None = None, event: Any = None) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        ...

    def _apply_event(self, event: Any) -> None:
        ...

    @callback
    def _handle_event(self, event: Any, device_id: str) -> None:
        ...
