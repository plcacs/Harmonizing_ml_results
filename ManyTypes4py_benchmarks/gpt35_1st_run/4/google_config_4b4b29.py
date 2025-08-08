from __future__ import annotations
import asyncio
from http import HTTPStatus
import logging
from typing import TYPE_CHECKING, Any
from hass_nabucasa import Cloud, cloud_api
from hass_nabucasa.google_report_state import ErrorResponse
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.google_assistant import DOMAIN as GOOGLE_DOMAIN
from homeassistant.components.google_assistant.helpers import AbstractConfig
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity, async_get_assistant_settings, async_get_entity_settings, async_listen_entity_updates, async_set_assistant_option, async_should_expose
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import CLOUD_NEVER_EXPOSED_ENTITIES
from homeassistant.core import CoreState, Event, HomeAssistant, State, callback, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, start
from homeassistant.helpers.entity import get_device_class
from homeassistant.helpers.entityfilter import EntityFilter
from homeassistant.setup import async_setup_component
from .const import CONF_ENTITY_CONFIG, CONF_FILTER, DEFAULT_DISABLE_2FA, DOMAIN as CLOUD_DOMAIN, PREF_DISABLE_2FA, PREF_SHOULD_EXPOSE
from .prefs import GOOGLE_SETTINGS_VERSION, CloudPreferences

if TYPE_CHECKING:
    from .client import CloudClient

_LOGGER: logging.Logger = logging.getLogger(__name__)
CLOUD_GOOGLE: str = f'{CLOUD_DOMAIN}.{GOOGLE_DOMAIN}'
SUPPORTED_DOMAINS: set[str] = {'alarm_control_panel', 'button', 'camera', 'climate', 'cover', 'fan', 'group', 'humidifier', 'input_boolean', 'input_button', 'input_select', 'light', 'lock', 'media_player', 'scene', 'script', 'select', 'switch', 'vacuum'}
SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES: set[BinarySensorDeviceClass] = {BinarySensorDeviceClass.DOOR, BinarySensorDeviceClass.GARAGE_DOOR, BinarySensorDeviceClass.LOCK, BinarySensorDeviceClass.MOTION, BinarySensorDeviceClass.OPENING, BinarySensorDeviceClass.PRESENCE, BinarySensorDeviceClass.WINDOW}
SUPPORTED_SENSOR_DEVICE_CLASSES: set[SensorDeviceClass] = {SensorDeviceClass.AQI, SensorDeviceClass.CO, SensorDeviceClass.CO2, SensorDeviceClass.HUMIDITY, SensorDeviceClass.PM10, SensorDeviceClass.PM25, SensorDeviceClass.TEMPERATURE, SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS}

def _supported_legacy(hass: HomeAssistant, entity_id: str) -> bool:
    ...

class CloudGoogleConfig(AbstractConfig):
    def __init__(self, hass: HomeAssistant, config: dict, cloud_user: Any, prefs: CloudPreferences, cloud: Cloud) -> None:
        ...

    @property
    def enabled(self) -> bool:
        ...

    @property
    def entity_config(self) -> dict:
        ...

    @property
    def secure_devices_pin(self) -> Any:
        ...

    @property
    def should_report_state(self) -> bool:
        ...

    def get_local_webhook_id(self, agent_user_id: Any) -> Any:
        ...

    def get_local_user_id(self, webhook_id: Any) -> Any:
        ...

    @property
    def cloud_user(self) -> Any:
        ...

    def _migrate_google_entity_settings_v1(self) -> None:
        ...

    async def async_initialize(self) -> None:
        ...

    def should_expose(self, state: State) -> bool:
        ...

    def _should_expose_legacy(self, entity_id: str) -> bool:
        ...

    def _should_expose_entity_id(self, entity_id: str) -> bool:
        ...

    @property
    def agent_user_id(self) -> str:
        ...

    @property
    def has_registered_user_agent(self) -> bool:
        ...

    def get_agent_user_id_from_context(self, context: Any) -> str:
        ...

    def get_agent_user_id_from_webhook(self, webhook_id: Any) -> Any:
        ...

    def _2fa_disabled_legacy(self, entity_id: str) -> Any:
        ...

    def should_2fa(self, state: State) -> bool:
        ...

    async def async_report_state(self, message: Any, agent_user_id: Any, event_id: Any = None) -> None:
        ...

    async def _async_request_sync_devices(self, agent_user_id: Any) -> HTTPStatus:
        ...

    async def async_connect_agent_user(self, agent_user_id: Any) -> None:
        ...

    async def async_disconnect_agent_user(self, agent_user_id: Any) -> None:
        ...

    @callback
    def async_get_agent_users(self) -> tuple:
        ...

    async def _async_prefs_updated(self, prefs: CloudPreferences) -> None:
        ...

    @callback
    def _async_exposed_entities_updated(self) -> None:
        ...

    @callback
    def _handle_entity_registry_updated(self, event: Event) -> None:
        ...

    @callback
    def _handle_device_registry_updated(self, event: Event) -> None:
        ...
