from typing import Any, Dict, List
from homeassistant.helpers.typing import ConfigType

def register_device(hass: HomeAssistant, api_key: str, name: str, device_id: str, device_ids: str, device_names: str) -> None:
def setup(hass: HomeAssistant, config: ConfigType) -> bool:
