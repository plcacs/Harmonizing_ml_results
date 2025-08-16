from __future__ import annotations
import asyncio
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, NamedTuple
from aioesphomeapi import APIClient, APIConnectionError, APIVersion, DeviceInfo as EsphomeDeviceInfo, EntityInfo, HomeassistantServiceCall, InvalidAuthAPIError, InvalidEncryptionKeyAPIError, ReconnectLogic, RequiresEncryptionAPIError, UserService, UserServiceArgType
from awesomeversion import AwesomeVersion
import voluptuous as vol
from homeassistant.components import bluetooth, tag, zeroconf
from homeassistant.const import ATTR_DEVICE_ID, CONF_MODE, EVENT_HOMEASSISTANT_CLOSE, EVENT_LOGGING_CHANGED, Platform
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, ServiceCall, State, callback
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv, device_registry as dr, template
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue, async_delete_issue
from homeassistant.helpers.service import async_set_service_schema
from homeassistant.helpers.template import Template
from homeassistant.util.async_ import create_eager_task
from .bluetooth import async_connect_scanner
from .const import CONF_ALLOW_SERVICE_CALLS, CONF_DEVICE_NAME, DEFAULT_ALLOW_SERVICE_CALLS, DEFAULT_URL, DOMAIN, PROJECT_URLS, STABLE_BLE_VERSION, STABLE_BLE_VERSION_STR
from .dashboard import async_get_dashboard
from .domain_data import DomainData
from .entry_data import ESPHomeConfigEntry, RuntimeEntryData
_LOGGER: logging.Logger

@callback
def _async_check_firmware_version(hass: HomeAssistant, device_info: EsphomeDeviceInfo, api_version: APIVersion) -> None:
    ...

@callback
def _async_check_using_api_password(hass: HomeAssistant, device_info: EsphomeDeviceInfo, has_password: bool) -> None:
    ...

class ESPHomeManager:
    def __init__(self, hass: HomeAssistant, entry: ESPHomeConfigEntry, host: str, password: str, cli: APIClient, zeroconf_instance: zeroconf.Zeroconf, domain_data: DomainData) -> None:
        ...

    async def on_stop(self, event: Event) -> None:
        ...

    @property
    def services_issue(self) -> str:
        ...

    @callback
    def async_on_service_call(self, service: HomeassistantServiceCall) -> None:
        ...

    @callback
    def _send_home_assistant_state(self, entity_id: str, attribute: str, state: State) -> None:
        ...

    @callback
    def _send_home_assistant_state_event(self, attribute: str, event: Event) -> None:
        ...

    @callback
    def async_on_state_subscription(self, entity_id: str, attribute: str = None) -> None:
        ...

    @callback
    def async_on_state_request(self, entity_id: str, attribute: str = None) -> None:
        ...

    async def on_connect(self) -> None:
        ...

    async def _on_connnect(self) -> None:
        ...

    async def on_disconnect(self, expected_disconnect: bool) -> None:
        ...

    async def on_connect_error(self, err: Exception) -> None:
        ...

    @callback
    def _async_handle_logging_changed(self, _event: Event) -> None:
        ...

    async def async_start(self) -> None:
        ...

@callback
def _async_setup_device_registry(hass: HomeAssistant, entry: ESPHomeConfigEntry, entry_data: RuntimeEntryData) -> None:
    ...

class ServiceMetadata(NamedTuple):
    ...

@callback
def execute_service(entry_data: RuntimeEntryData, service: UserService, call: ServiceCall) -> None:
    ...

def build_service_name(device_info: EsphomeDeviceInfo, service: UserService) -> str:
    ...

@callback
def _async_register_service(hass: HomeAssistant, entry_data: RuntimeEntryData, device_info: EsphomeDeviceInfo, service: UserService) -> None:
    ...

@callback
def _setup_services(hass: HomeAssistant, entry_data: RuntimeEntryData, services: list[UserService]) -> None:
    ...

async def cleanup_instance(hass: HomeAssistant, entry: ESPHomeConfigEntry) -> RuntimeEntryData:
    ...
