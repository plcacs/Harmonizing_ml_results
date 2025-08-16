def setup(hass: HomeAssistant, config: ConfigType) -> bool:
def _system_callback_handler(hass: HomeAssistant, config: ConfigType, src: str, *args) -> None:
def _get_devices(hass: HomeAssistant, discovery_type: str, keys: List[str], interface: str) -> List[Dict[str, Any]]:
def _create_ha_id(name: str, channel: int, param: Optional[str], count: int) -> str:
def _hm_event_handler(hass: HomeAssistant, interface: str, device: str, caller: str, attribute: str, value: Any) -> None:
def _device_from_servicecall(hass: HomeAssistant, service: ServiceCall) -> Optional[Any]:
