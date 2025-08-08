async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class SatelIntegraSwitch(SwitchEntity):
    def __init__(self, controller: Any, device_number: int, device_name: str, code: str) -> None:

    async def async_added_to_hass(self) -> None:

    def _devices_updated(self, zones: list[int]) -> None:

    async def async_turn_on(self, **kwargs) -> None:

    async def async_turn_off(self, **kwargs) -> None:

    @property
    def is_on(self) -> bool:

    def _read_state(self) -> bool:

    @property
    def name(self) -> str:
