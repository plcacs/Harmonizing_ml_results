"""Stub file for google_config_4b4b29 module."""

from __future__ import annotations
from collections.abc import Callable
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.google_assistant import DOMAIN as GOOGLE_DOMAIN
from homeassistant.components.google_assistant.helpers import AbstractConfig
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
    async_get_assistant_settings,
    async_get_entity_settings,
    async_listen_entity_updates,
    async_set_assistant_option,
    async_should_expose,
)
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import (
    CoreState,
    Event,
    HomeAssistant,
    State,
    callback,
    split_entity_id,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entityfilter import EntityFilter
from hass_nabucasa import Cloud, CloudPreferences
from .const import (
    CONF_ENTITY_CONFIG,
    CONF_FILTER,
    DEFAULT_DISABLE_2FA,
    DOMAIN as CLOUD_DOMAIN,
    PREF_DISABLE_2FA,
    PREF_SHOULD_EXPOSE,
)
from .prefs import (
    GOOGLE_SETTINGS_VERSION,
    CloudPreferences,
)

SUPPORTED_DOMAINS: set[str] = ...
SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES: set[BinarySensorDeviceClass] = ...
SUPPORTED_SENSOR_DEVICE_CLASSES: set[SensorDeviceClass] = ...

def _supported_legacy(hass: HomeAssistant, entity_id: str) -> bool:
    ...

class CloudGoogleConfig(AbstractConfig):
    """HA Cloud Configuration for Google Assistant."""

    def __init__(
        self,
        hass: HomeAssistant,
        config: Dict[str, Any],
        cloud_user: str,
        prefs: CloudPreferences,
        cloud: Cloud,
    ) -> None:
        ...

    @property
    def enabled(self) -> bool:
        ...

    @property
    def entity_config(self) -> Dict[str, Any]:
        ...

    @property
    def secure_devices_pin(self) -> str:
        ...

    @property
    def should_report_state(self) -> bool:
        ...

    def get_local_webhook_id(self, agent_user_id: str) -> str:
        ...

    def get_local_user_id(self, webhook_id: str) -> str:
        ...

    @property
    def cloud_user(self) -> str:
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

    def get_agent_user_id_from_webhook(self, webhook_id: str) -> Optional[str]:
        ...

    def _2fa_disabled_legacy(self, entity_id: str) -> Optional[bool]:
        ...

    def should_2fa(self, state: State) -> bool:
        ...

    async def async_report_state(
        self,
        message: Any,
        agent_user_id: str,
        event_id: Optional[str] = None,
    ) -> None:
        ...

    async def _async_request_sync_devices(self, agent_user_id: str) -> HTTPStatus:
        ...

    async def async_connect_agent_user(self, agent_user_id: str) -> None:
        ...

    async def async_disconnect_agent_user(self, agent_user_id: str) -> None:
        ...

    @callback
    def async_get_agent_users(self) -> Tuple[str, ...]:
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