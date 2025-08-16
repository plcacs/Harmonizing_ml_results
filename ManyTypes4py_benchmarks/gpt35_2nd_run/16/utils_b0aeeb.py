def add_insteon_events(hass: HomeAssistant, device: Device) -> None:
def register_new_device_callback(hass: HomeAssistant) -> None:
def async_register_services(hass: HomeAssistant) -> None:
def print_aldb_to_log(aldb: Any) -> None:
def async_add_insteon_entities(hass: HomeAssistant, platform: Platform, entity_type: Callable, async_add_entities: AddConfigEntryEntitiesCallback, discovery_info: dict) -> None:
def async_add_insteon_devices(hass: HomeAssistant, platform: Platform, entity_type: Callable, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
def async_get_usb_ports(hass: HomeAssistant) -> Any:
def compute_device_name(ha_device: Any) -> str:
async def async_device_name(dev_registry: Any, address: Address) -> str:
