async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class DemoLock(LockEntity):
    _attr_should_poll: bool = False

    def __init__(self, name: str, state: LockState, openable: bool = False, jam_on_operation: bool = False) -> None:
        ...

    @property
    def is_locking(self) -> bool:
        ...

    @property
    def is_unlocking(self) -> bool:
        ...

    @property
    def is_jammed(self) -> bool:
        ...

    @property
    def is_locked(self) -> bool:
        ...

    @property
    def is_open(self) -> bool:
        ...

    @property
    def is_opening(self) -> bool:
        ...

    async def async_lock(self, **kwargs: Any) -> None:
        ...

    async def async_unlock(self, **kwargs: Any) -> None:
        ...

    async def async_open(self, **kwargs: Any) -> None:
        ...
