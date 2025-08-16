from __future__ import annotations
import asyncio
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta
from http import HTTPStatus
import logging
from typing import TYPE_CHECKING, Any
import aiohttp
from hass_nabucasa import Cloud, cloud_api
from yarl import URL
from homeassistant.components import persistent_notification
from homeassistant.components.alexa import DOMAIN as ALEXA_DOMAIN, config as alexa_config, entities as alexa_entities, errors as alexa_errors, state_report as alexa_state_report
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity, async_get_assistant_settings, async_listen_entity_updates, async_should_expose
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import CLOUD_NEVER_EXPOSED_ENTITIES
from homeassistant.core import Event, HomeAssistant, callback, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, start
from homeassistant.helpers.entity import get_device_class
from homeassistant.helpers.entityfilter import EntityFilter
from homeassistant.helpers.event import async_call_later
from homeassistant.setup import async_setup_component
from homeassistant.util.dt import utcnow
from .const import CONF_ENTITY_CONFIG, CONF_FILTER, DOMAIN as CLOUD_DOMAIN, PREF_ALEXA_REPORT_STATE, PREF_ENABLE_ALEXA, PREF_SHOULD_EXPOSE
from .prefs import ALEXA_SETTINGS_VERSION, CloudPreferences
if TYPE_CHECKING:
    from .client import CloudClient
_LOGGER: logging.Logger = logging.getLogger(__name__)
CLOUD_ALEXA: str = f'{CLOUD_DOMAIN}.{ALEXA_DOMAIN}'
SYNC_DELAY: int = 1
SUPPORTED_DOMAINS: set[str] = {'alarm_control_panel', 'alert', 'automation', 'button', 'camera', 'climate', 'cover', 'fan', 'group', 'humidifier', 'image_processing', 'input_boolean', 'input_button', 'input_number', 'light', 'lock', 'media_player', 'number', 'scene', 'script', 'switch', 'timer', 'vacuum'}
SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES: set[BinarySensorDeviceClass] = {BinarySensorDeviceClass.DOOR, BinarySensorDeviceClass.GARAGE_DOOR, BinarySensorDeviceClass.MOTION, BinarySensorDeviceClass.OPENING, BinarySensorDeviceClass.PRESENCE, BinarySensorDeviceClass.WINDOW}
SUPPORTED_SENSOR_DEVICE_CLASSES: set[SensorDeviceClass] = {SensorDeviceClass.TEMPERATURE}

def entity_supported(hass: HomeAssistant, entity_id: str) -> bool:
    """Return if the entity is supported.

    This is called when migrating from legacy config format to avoid exposing
    all binary sensors and sensors.
    """
    domain: str = split_entity_id(entity_id)[0]
    if domain in SUPPORTED_DOMAINS:
        return True
    try:
        device_class: str = get_device_class(hass, entity_id)
    except HomeAssistantError:
        return False
    if domain == 'binary_sensor' and device_class in SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES:
        return True
    if domain == 'sensor' and device_class in SUPPORTED_SENSOR_DEVICE_CLASSES:
        return True
    return False

class CloudAlexaConfig(alexa_config.AbstractConfig):
    """Alexa Configuration."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any], cloud_user: str, prefs: CloudPreferences, cloud: Cloud):
        """Initialize the Alexa config."""
        super().__init__(hass)
        self._config: dict[str, Any] = config
        self._cloud_user: str = cloud_user
        self._prefs: CloudPreferences = prefs
        self._cloud: Cloud = cloud
        self._token: str | None = None
        self._token_valid: datetime | None = None
        self._cur_entity_prefs: dict[str, Any] = async_get_assistant_settings(hass, CLOUD_ALEXA)
        self._alexa_sync_unsub: asyncio.TimerHandle | None = None
        self._endpoint: str | None = None

    @property
    def enabled(self) -> bool:
        """Return if Alexa is enabled."""
        return self._cloud.is_logged_in and (not self._cloud.subscription_expired) and self._prefs.alexa_enabled

    @property
    def supports_auth(self) -> bool:
        """Return if config supports auth."""
        return True

    @property
    def should_report_state(self) -> bool:
        """Return if states should be proactively reported."""
        return self._prefs.alexa_enabled and self._prefs.alexa_report_state and self.authorized

    @property
    def endpoint(self) -> str:
        """Endpoint for report state."""
        if self._endpoint is None:
            raise ValueError('No endpoint available. Fetch access token first')
        return self._endpoint

    @property
    def locale(self) -> str:
        """Return config locale."""
        return 'en-US'

    @property
    def entity_config(self) -> dict[str, Any]:
        """Return entity config."""
        return self._config.get(CONF_ENTITY_CONFIG) or {}

    @callback
    def user_identifier(self) -> str:
        """Return an identifier for the user that represents this config."""
        return self._cloud_user
