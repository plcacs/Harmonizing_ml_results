async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

    def __init__(self, peripheral: Any, parent_name: str, unit: str, measurement: str, consumer: Any) -> None:

    @property
    def unique_id(self) -> str:

    @property
    def name(self) -> str:

    @property
    def native_value(self) -> Any:

    @property
    def native_unit_of_measurement(self) -> str:

    @property
    def available(self) -> bool:

    async def async_update(self) -> None:
