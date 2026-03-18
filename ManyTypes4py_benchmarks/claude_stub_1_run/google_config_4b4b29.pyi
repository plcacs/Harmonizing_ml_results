```pyi
from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import Any, Callable

from hass_nabucasa import Cloud
from homeassistant.components.google_assistant.helpers import AbstractConfig
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers.entityfilter import EntityFilter

from .prefs import CloudPreferences

_LOGGER: Any

CLOUD_GOOGLE: str
SUPPORTED_DOMAINS: set[str]
SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES: set[Any]
SUPPORTED_SENSOR_DEVICE_CLASSES: set[Any]

def _supported_legacy(hass: HomeAssistant, entity_id: str) -> bool: ...

class CloudGoogleConfig(AbstractConfig):
    _config: dict[str, Any]
    _user: Any
    _prefs: CloudPreferences
    _cloud: Cloud
    _sync_entities_lock: asyncio.Lock
    
    def __init__(
        self,
        hass: HomeAssistant,
        config: dict[str, Any],
        cloud_user: Any,
        prefs: CloudPreferences,
        cloud: Cloud,
    ) -> None: ...
    
    @property
    def enabled(self) -> bool: ...
    
    @property
    def entity_config(self) -> dict[str, Any]: ...
    
    @property
    def secure_devices_pin(self) -> str | None: ...
    
    @property
    def should_report_state(self) -> bool: ...
    
    def get_local_webhook_id(self, agent_user_id: str) -> str | None: ...
    
    def get_local_user_id(self, webhook_id: str) -> Any: ...
    
    @property
    def cloud_user(self) -> Any: ...
    
    def _migrate_google_entity_settings_v1(self) -> None: ...
    
    async def async_initialize(self) -> None: ...
    
    def should_expose(self, state: State) -> bool: ...
    
    def _should_expose_legacy(self, entity_id: str) -> bool: ...
    
    def _should_expose_entity_id(self, entity_id: str) -> bool: ...
    
    @property
    def agent_user_id(self) -> str | None: ...
    
    @property
    def has_registered_user_agent(self) -> bool: ...
    
    def get_agent_user_id_from_context(self, context: Any) -> str | None: ...
    
    def get_agent_user_id_from_webhook(self, webhook_id: str) -> str | None: ...
    
    def _2fa_disabled_legacy(self, entity_id: str) -> bool | None: ...
    
    def should_2fa(self, state: State) -> bool: ...
    
    async def async_report_state(
        self,
        message: dict[str, Any],
        agent_user_id: str,
        event_id: str | None = None,
    ) -> None: ...
    
    async def _async_request_sync_devices(self, agent_user_id: str) -> HTTPStatus: ...
    
    async def async_connect_agent_user(self, agent_user_id: str) -> None: ...
    
    async def async_disconnect_agent_user(self, agent_user_id: str) -> None: ...
    
    @callback
    def async_get_agent_users(self) -> tuple[str, ...]: ...
    
    async def _async_prefs_updated(self, prefs: CloudPreferences) -> None: ...
    
    @callback
    def _async_exposed_entities_updated(self) -> None: ...
    
    @callback
    def _handle_entity_registry_updated(self, event: Event) -> None: ...
    
    @callback
    def _handle_device_registry_updated(self, event: Event) -> None: ...
```