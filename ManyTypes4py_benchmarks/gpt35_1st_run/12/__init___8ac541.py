import logging
from typing import List, Dict, Any
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ServiceCall
import voluptuous as vol

_LOGGER = logging.getLogger(__name__)
DOMAIN: str = 'joaoapps_join'
CONF_API_KEY: str = 'api_key'
CONF_DEVICE_ID: str = 'device_id'
CONF_NAME: str = 'name'
CONF_DEVICE_IDS: str = 'device_ids'
CONF_DEVICE_NAMES: str = 'device_names'
CONFIG_SCHEMA: Dict[str, Any] = vol.Schema({
    DOMAIN: vol.All(
        cv.ensure_list, [{
            vol.Required(CONF_API_KEY): cv.string,
            vol.Optional(CONF_DEVICE_ID): cv.string,
            vol.Optional(CONF_DEVICE_IDS): cv.string,
            vol.Optional(CONF_DEVICE_NAMES): cv.string,
            vol.Optional(CONF_NAME): cv.string
        }]
    )
}, extra=vol.ALLOW_EXTRA)

def register_device(hass: HomeAssistant, api_key: str, name: str, device_id: str, device_ids: str, device_names: str) -> None:
    def ring_service(service: ServiceCall) -> None:
        ...

    def set_wallpaper_service(service: ServiceCall) -> None:
        ...

    def send_file_service(service: ServiceCall) -> None:
        ...

    def send_url_service(service: ServiceCall) -> None:
        ...

    def send_tasker_service(service: ServiceCall) -> None:
        ...

    def send_sms_service(service: ServiceCall) -> None:
        ...

def setup(hass: HomeAssistant, config: Dict[str, Any]) -> bool:
    ...
