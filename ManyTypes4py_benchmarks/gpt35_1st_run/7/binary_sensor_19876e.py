def no_missing_threshold(value: Mapping[str, Any]) -> Mapping[str, Any]:
async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
def _threshold_type(lower: float, upper: float) -> str:
class ThresholdSensor(BinarySensorEntity):
    def __init__(self, entity_id: str, name: str, lower: float, upper: float, hysteresis: float, device_class: str, unique_id: str, device_info: DeviceInfo = None) -> None:
    async def async_added_to_hass(self) -> None:
    def _async_setup_sensor(self) -> None:
    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
    def _update_state(self) -> None:
    def below(sensor_value: float, threshold: float) -> bool:
    def above(sensor_value: float, threshold: float) -> bool:
    @callback
    def async_start_preview(self, preview_callback: Callable) -> CALLBACK_TYPE:
