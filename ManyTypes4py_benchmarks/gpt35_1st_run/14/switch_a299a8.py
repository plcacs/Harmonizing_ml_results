async def async_setup_entry(hass: HomeAssistant, config_entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

def __init__(self, smappee_base: Any, service_location: Any, name: str, actuator_id: str, actuator_type: str, actuator_serialnumber: str, actuator_state_option: Any = None) -> None:

@property
def name(self) -> str:

@property
def is_on(self) -> bool:

def turn_on(self, **kwargs: Any) -> None:

def turn_off(self, **kwargs: Any) -> None:

@property
def available(self) -> bool:

@property
def unique_id(self) -> str:

async def async_update(self) -> None:
