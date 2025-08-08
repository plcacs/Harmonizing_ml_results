def log_entry(hass: HomeAssistant, name: str, message: str, domain: str = None, entity_id: str = None, context: Context = None) -> None:

def async_log_entry(hass: HomeAssistant, name: str, message: str, domain: str = None, entity_id: str = None, context: Context = None) -> None:

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:

def log_message(service: ServiceCall) -> None:

def _process_logbook_platform(hass: HomeAssistant, domain: str, platform: Callable) -> None:

def _async_describe_event(domain: str, event_name: str, describe_callback: Callable) -> None:
