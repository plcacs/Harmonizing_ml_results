def async_setup_entry_base(hass: HomeAssistant, config_entry: DynaliteConfigEntry, async_add_entities: AddEntitiesCallback, platform: str, entity_from_device: Callable[[Any, DynaliteBridge], DynaliteBase]) -> None:

def __init__(self, device: Any, bridge: DynaliteBridge) -> None:

def unique_id(self) -> str:

def available(self) -> bool:

def device_info(self) -> DeviceInfo:

async def async_added_to_hass(self) -> None:

def initialize_state(self, state: Any) -> None:

async def async_will_remove_from_hass(self) -> None:
