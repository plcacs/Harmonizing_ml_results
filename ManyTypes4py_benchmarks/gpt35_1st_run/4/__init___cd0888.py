from typing import Any, Dict, Set, Tuple

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
async def async_new_client(hass: HomeAssistant, session: Session, entry: ConfigEntry) -> None:
async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
class TelldusLiveClient:
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, session: Session, interval: int) -> None:
    async def async_get_hubs(self) -> List[Dict[str, Any]]:
    def device_info(self, device_id: str) -> Dict[str, Any]:
    @staticmethod
    def identify_device(device: Any) -> str:
    async def _discover(self, device_id: str) -> None:
    async def update(self, *args: Any) -> None:
    def device(self, device_id: str) -> Any:
    def is_available(self, device_id: str) -> bool:
