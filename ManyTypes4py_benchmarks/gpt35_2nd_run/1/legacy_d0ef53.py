from __future__ import annotations
import asyncio
from collections.abc import Callable, Coroutine, Sequence
from datetime import datetime, timedelta
import hashlib
from types import ModuleType
from typing import Any, Final, Protocol, final
import attr
from propcache.api import cached_property
import voluptuous as vol
from homeassistant import util
from homeassistant.components import zone
from homeassistant.components.zone import ENTITY_ID_HOME
from homeassistant.config import async_log_schema_error, config_per_platform, load_yaml_config_file
from homeassistant.const import ATTR_ENTITY_ID, ATTR_GPS_ACCURACY, ATTR_ICON, ATTR_LATITUDE, ATTR_LONGITUDE, ATTR_NAME, CONF_ICON, CONF_MAC, CONF_NAME, DEVICE_DEFAULT_NAME, EVENT_HOMEASSISTANT_STOP, STATE_HOME, STATE_NOT_HOME
from homeassistant.core import Event, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, discovery, entity_registry as er
from homeassistant.helpers.event import async_track_time_interval, async_track_utc_time_change
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, GPSType, StateType
from homeassistant.setup import SetupPhases, async_notify_setup_error, async_prepare_setup_platform, async_start_setup
from homeassistant.util import dt as dt_util
from homeassistant.util.async_ import create_eager_task
from homeassistant.util.yaml import dump
from .const import ATTR_ATTRIBUTES, ATTR_BATTERY, ATTR_CONSIDER_HOME, ATTR_DEV_ID, ATTR_GPS, ATTR_HOST_NAME, ATTR_LOCATION_NAME, ATTR_MAC, ATTR_SOURCE_TYPE, CONF_CONSIDER_HOME, CONF_NEW_DEVICE_DEFAULTS, CONF_SCAN_INTERVAL, CONF_TRACK_NEW, DEFAULT_CONSIDER_HOME, DEFAULT_TRACK_NEW, DOMAIN, LOGGER, PLATFORM_TYPE_LEGACY, SCAN_INTERVAL, SourceType
SERVICE_SEE: Final = 'see'
SOURCE_TYPES: Final = [cls.value for cls in SourceType]
NEW_DEVICE_DEFAULTS_SCHEMA: Final = vol.Any(None, vol.Schema({vol.Optional(CONF_TRACK_NEW, default=DEFAULT_TRACK_NEW): cv.boolean}))
PLATFORM_SCHEMA: Final = cv.PLATFORM_SCHEMA.extend({vol.Optional(CONF_SCAN_INTERVAL): cv.time_period, vol.Optional(CONF_TRACK_NEW): cv.boolean, vol.Optional(CONF_CONSIDER_HOME, default=DEFAULT_CONSIDER_HOME): vol.All(cv.time_period, cv.positive_timedelta), vol.Optional(CONF_NEW_DEVICE_DEFAULTS, default={}): NEW_DEVICE_DEFAULTS_SCHEMA})
PLATFORM_SCHEMA_BASE: Final = cv.PLATFORM_SCHEMA_BASE.extend(PLATFORM_SCHEMA.schema)
SERVICE_SEE_PAYLOAD_SCHEMA: Final = vol.Schema(vol.All(cv.has_at_least_one_key(ATTR_MAC, ATTR_DEV_ID), {ATTR_MAC: cv.string, ATTR_DEV_ID: cv.string, ATTR_HOST_NAME: cv.string, ATTR_LOCATION_NAME: cv.string, ATTR_GPS: cv.gps, ATTR_GPS_ACCURACY: cv.positive_int, ATTR_BATTERY: cv.positive_int, ATTR_ATTRIBUTES: dict, ATTR_SOURCE_TYPE: vol.Coerce(SourceType), ATTR_CONSIDER_HOME: cv.time_period, vol.Optional('battery_status'): str, vol.Optional('hostname'): str}))
YAML_DEVICES: Final = 'known_devices.yaml'
EVENT_NEW_DEVICE: Final = 'device_tracker_new_device'

class SeeCallback(Protocol):
    def __call__(self, mac=None, dev_id=None, host_name=None, location_name=None, gps=None, gps_accuracy=None, battery=None, attributes=None, source_type=SourceType.GPS, picture=None, icon=None, consider_home=None) -> None: ...

class AsyncSeeCallback(Protocol):
    async def __call__(self, mac=None, dev_id=None, host_name=None, location_name=None, gps=None, gps_accuracy=None, battery=None, attributes=None, source_type=SourceType.GPS, picture=None, icon=None, consider_home=None) -> None: ...

def see(hass: HomeAssistant, mac=None, dev_id=None, host_name=None, location_name=None, gps=None, gps_accuracy=None, battery=None, attributes=None) -> None: ...

@callback
def async_setup_integration(hass: HomeAssistant, config: ConfigType) -> None: ...

async def _async_setup_integration(hass: HomeAssistant, config: ConfigType, tracker_future: asyncio.Future) -> None: ...

@attr.s
class DeviceTrackerPlatform:
    name: str = attr.ib()
    platform: ModuleType = attr.ib()
    config: ConfigType = attr.ib()

    @cached_property
    def type(self) -> str: ...

    async def async_setup_legacy(self, hass: HomeAssistant, tracker: DeviceTracker, discovery_info=None) -> None: ...

async def async_extract_config(hass: HomeAssistant, config: ConfigType) -> list: ...

async def async_create_platform_type(hass: HomeAssistant, config: ConfigType, p_type: str, p_config: ConfigType) -> DeviceTrackerPlatform: ...

def _load_device_names_and_attributes(scanner, device_name_uses_executor, extra_attributes_uses_executor, seen, found_devices) -> tuple: ...

@callback
def async_setup_scanner_platform(hass: HomeAssistant, config: ConfigType, scanner, async_see_device, platform) -> None: ...

async def get_tracker(hass: HomeAssistant, config: ConfigType) -> DeviceTracker: ...

class DeviceTracker:
    def __init__(self, hass: HomeAssistant, consider_home: timedelta, track_new: bool, defaults: dict, devices: list) -> None: ...

    def see(self, mac=None, dev_id=None, host_name=None, location_name=None, gps=None, gps_accuracy=None, battery=None, attributes=None, source_type=SourceType.GPS, picture=None, icon=None, consider_home=None) -> None: ...

    async def async_see(self, mac=None, dev_id=None, host_name=None, location_name=None, gps=None, gps_accuracy=None, battery=None, attributes=None, source_type=SourceType.GPS, picture=None, icon=None, consider_home=None) -> None: ...

    async def async_update_config(self, path: str, dev_id: str, device: Device) -> None: ...

    @callback
    def async_update_stale(self, now: datetime) -> None: ...

    async def async_setup_tracked_device(self) -> None: ...

class Device(RestoreEntity):
    def __init__(self, hass: HomeAssistant, consider_home: timedelta, track: bool, dev_id: str, mac: str, name=None, picture=None, gravatar=None, icon=None) -> None: ...

    async def async_seen(self, host_name=None, location_name=None, gps=None, gps_accuracy=None, battery=None, attributes=None, source_type=SourceType.GPS, consider_home=None) -> None: ...

    def stale(self, now=None) -> bool: ...

    def mark_stale(self) -> None: ...

    async def async_update(self) -> None: ...

    async def async_added_to_hass(self) -> None: ...

class DeviceScanner:
    def scan_devices(self) -> list: ...

    async def async_scan_devices(self) -> list: ...

    def get_device_name(self, device) -> str: ...

    async def async_get_device_name(self, device) -> str: ...

    def get_extra_attributes(self, device) -> dict: ...

    async def async_get_extra_attributes(self, device) -> dict: ...

async def async_load_config(path: str, hass: HomeAssistant, consider_home: timedelta) -> list: ...

def update_config(path: str, dev_id: str, device: Device) -> None: ...

def remove_device_from_config(hass: HomeAssistant, device_id: str) -> None: ...

def get_gravatar_for_email(email: str) -> str: ...
