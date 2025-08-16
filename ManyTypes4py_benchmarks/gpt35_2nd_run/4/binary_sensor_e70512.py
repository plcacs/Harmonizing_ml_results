async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class EnvisalinkBinarySensor(EnvisalinkEntity, BinarySensorEntity):
    def __init__(self, hass: HomeAssistant, zone_number: int, zone_name: str, zone_type: str, info: dict, controller: dict) -> None:

    async def async_added_to_hass(self) -> None:

    @property
    def extra_state_attributes(self) -> dict:

    @property
    def is_on(self) -> bool:

    @property
    def device_class(self) -> str:

    @callback
    def async_update_callback(self, zone: str) -> None:
