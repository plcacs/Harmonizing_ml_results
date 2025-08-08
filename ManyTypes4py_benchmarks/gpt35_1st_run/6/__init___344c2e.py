def _has_unique_names(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
AMCREST_SCHEMA: vol.Schema = vol.Schema({
CONFIG_SCHEMA: vol.Schema = vol.Schema({
class AmcrestChecker(ApiWrapper):
    def __init__(self, hass: HomeAssistant, name: str, host: str, port: int, user: str, password: str) -> None:
    def _handle_offline(self, ex: LoginError) -> None:
    def _async_handle_offline(self, ex: LoginError) -> None:
    def _handle_error(self) -> None:
    def _async_handle_error(self) -> None:
    def _set_online(self) -> None:
    def _async_set_online(self) -> None:
    def _async_signal_online(self) -> None:
    async def _wrap_test_online(self, now: datetime) -> None:
def _monitor_events(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: set[str]) -> None:
def _start_event_monitor(hass: HomeAssistant, name: str, api: AmcrestChecker, event_codes: set[str]) -> None:
async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
def have_permission(user: User, entity_id: str) -> bool:
async def async_extract_from_service(call: ServiceCall) -> list[str]:
async def async_service_handler(call: ServiceCall) -> None:
@dataclass
class AmcrestDevice:
