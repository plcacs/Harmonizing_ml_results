async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class VActuator(SwitchEntity):
    def __init__(self, peripheral: Any, parent_name: str, unit: str, measurement: str, consumer: Any) -> None:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def available(self) -> bool:
        ...

    async def async_turn_off(self, **kwargs) -> None:
        ...

    async def async_turn_on(self, **kwargs) -> None:
        ...

    async def update_state(self, state: int) -> None:
        ...

    async def async_update(self) -> None:
        ...
