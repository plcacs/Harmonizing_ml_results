def async_setup_entry_base(hass: HomeAssistant, config_entry: DynaliteConfigEntry, async_add_entities: AddEntitiesCallback, platform: str, entity_from_device: Callable[[Any, DynaliteBridge], DynaliteBase]) -> None:
    ...

class DynaliteBase(RestoreEntity, ABC):
    _attr_has_entity_name: bool = True
    _attr_name: str = None

    def __init__(self, device: Any, bridge: DynaliteBridge) -> None:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def device_info(self) -> DeviceInfo:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @abstractmethod
    def initialize_state(self, state: Any) -> None:

    async def async_will_remove_from_hass(self) -> None:
        ...
