"""Legacy device tracker classes."""
from __future__ import annotations
import asyncio
from collections.abc import Callable, Coroutine, Sequence, Mapping
from datetime import datetime, timedelta
import hashlib
from types import ModuleType
from typing import Any, Final, Protocol, final, Optional, Union, Dict, List, Set, Tuple, cast
import attr
from propcache.api import cached_property
import voluptuous as vol
from homeassistant import util
from homeassistant.components import zone
from homeassistant.components.zone import ENTITY_ID_HOME
from homeassistant.config import async_log_schema_error, config_per_platform, load_yaml_config_file
from homeassistant.const import ATTR_ENTITY_ID, ATTR_GPS_ACCURACY, ATTR_LATITUDE, ATTR_LONGITUDE, ATTR_NAME, CONF_ICON, CONF_MAC, CONF_NAME, DEVICE_DEFAULT_NAME, EVENT_HOMEASSISTANT_STOP, STATE_HOME, STATE_NOT_HOME
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
SOURCE_TYPES: Final[List[str]] = [cls.value for cls in SourceType]
NEW_DEVICE_DEFAULTS_SCHEMA: Final = vol.Any(None, vol.Schema({vol.Optional(CONF_TRACK_NEW, default=DEFAULT_TRACK_NEW): cv.boolean}))
PLATFORM_SCHEMA: Final = cv.PLATFORM_SCHEMA.extend({vol.Optional(CONF_SCAN_INTERVAL): cv.time_period, vol.Optional(CONF_TRACK_NEW): cv.boolean, vol.Optional(CONF_CONSIDER_HOME, default=DEFAULT_CONSIDER_HOME): vol.All(cv.time_period, cv.positive_timedelta), vol.Optional(CONF_NEW_DEVICE_DEFAULTS, default={}): NEW_DEVICE_DEFAULTS_SCHEMA})
PLATFORM_SCHEMA_BASE: Final = cv.PLATFORM_SCHEMA_BASE.extend(PLATFORM_SCHEMA.schema)
SERVICE_SEE_PAYLOAD_SCHEMA: Final = vol.Schema(vol.All(cv.has_at_least_one_key(ATTR_MAC, ATTR_DEV_ID), {ATTR_MAC: cv.string, ATTR_DEV_ID: cv.string, ATTR_HOST_NAME: cv.string, ATTR_LOCATION_NAME: cv.string, ATTR_GPS: cv.gps, ATTR_GPS_ACCURACY: cv.positive_int, ATTR_BATTERY: cv.positive_int, ATTR_ATTRIBUTES: dict, ATTR_SOURCE_TYPE: vol.Coerce(SourceType), ATTR_CONSIDER_HOME: cv.time_period, vol.Optional('battery_status'): str, vol.Optional('hostname'): str}))
YAML_DEVICES: Final = 'known_devices.yaml'
EVENT_NEW_DEVICE: Final = 'device_tracker_new_device'

class SeeCallback(Protocol):
    """Protocol type for DeviceTracker.see callback."""
    def __call__(self, mac: Optional[str] = None, dev_id: Optional[str] = None, host_name: Optional[str] = None, location_name: Optional[str] = None, gps: Optional[GPSType] = None, gps_accuracy: Optional[int] = None, battery: Optional[int] = None, attributes: Optional[Dict[str, Any]] = None, source_type: SourceType = SourceType.GPS, picture: Optional[str] = None, icon: Optional[str] = None, consider_home: Optional[timedelta] = None) -> None:
        """Define see type."""

class AsyncSeeCallback(Protocol):
    """Protocol type for DeviceTracker.async_see callback."""
    async def __call__(self, mac: Optional[str] = None, dev_id: Optional[str] = None, host_name: Optional[str] = None, location_name: Optional[str] = None, gps: Optional[GPSType] = None, gps_accuracy: Optional[int] = None, battery: Optional[int] = None, attributes: Optional[Dict[str, Any]] = None, source_type: SourceType = SourceType.GPS, picture: Optional[str] = None, icon: Optional[str] = None, consider_home: Optional[timedelta] = None) -> None:
        """Define async_see type."""

def see(hass: HomeAssistant, mac: Optional[str] = None, dev_id: Optional[str] = None, host_name: Optional[str] = None, location_name: Optional[str] = None, gps: Optional[GPSType] = None, gps_accuracy: Optional[int] = None, battery: Optional[int] = None, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Call service to notify you see device."""
    data: Dict[str, Any] = {key: value for key, value in ((ATTR_MAC, mac), (ATTR_DEV_ID, dev_id), (ATTR_HOST_NAME, host_name), (ATTR_LOCATION_NAME, location_name), (ATTR_GPS, gps), (ATTR_GPS_ACCURACY, gps_accuracy), (ATTR_BATTERY, battery)) if value is not None}
    if attributes is not None:
        data[ATTR_ATTRIBUTES] = attributes
    hass.services.call(DOMAIN, SERVICE_SEE, data)

@callback
def async_setup_integration(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up the legacy integration."""
    tracker_future = hass.loop.create_future()

    async def async_platform_discovered(p_type: str, info: Dict[str, Any]) -> None:
        """Load a platform."""
        platform = await async_create_platform_type(hass, config, p_type, {})
        if platform is None or platform.type != PLATFORM_TYPE_LEGACY:
            return
        tracker = await tracker_future
        await platform.async_setup_legacy(hass, tracker, info)
    discovery.async_listen_platform(hass, DOMAIN, async_platform_discovered)
    hass.async_create_task(_async_setup_integration(hass, config, tracker_future), eager_start=True)

async def _async_setup_integration(hass: HomeAssistant, config: ConfigType, tracker_future: asyncio.Future) -> None:
    """Set up the legacy integration."""
    tracker = await get_tracker(hass, config)
    tracker_future.set_result(tracker)

    async def async_see_service(call: ServiceCall) -> None:
        """Service to see a device."""
        data = dict(call.data)
        data.pop('hostname', None)
        data.pop('battery_status', None)
        await tracker.async_see(**data)
    hass.services.async_register(DOMAIN, SERVICE_SEE, async_see_service, SERVICE_SEE_PAYLOAD_SCHEMA)
    legacy_platforms = await async_extract_config(hass, config)
    setup_tasks = [create_eager_task(legacy_platform.async_setup_legacy(hass, tracker)) for legacy_platform in legacy_platforms]
    if setup_tasks:
        await asyncio.wait(setup_tasks)
    cancel_update_stale = async_track_utc_time_change(hass, tracker.async_update_stale, second=range(0, 60, 5))
    await tracker.async_setup_tracked_device()

    @callback
    def _on_hass_stop(_: Event) -> None:
        """Cleanup when Home Assistant stops."""
        cancel_update_stale()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _on_hass_stop)

@attr.s
class DeviceTrackerPlatform:
    """Class to hold platform information."""
    LEGACY_SETUP: Final = ('async_get_scanner', 'get_scanner', 'async_setup_scanner', 'setup_scanner')
    name: str = attr.ib()
    platform: ModuleType = attr.ib()
    config: ConfigType = attr.ib()

    @cached_property
    def type(self) -> Optional[str]:
        """Return platform type."""
        methods, platform_type = (self.LEGACY_SETUP, PLATFORM_TYPE_LEGACY)
        for method in methods:
            if hasattr(self.platform, method):
                return platform_type
        return None

    async def async_setup_legacy(self, hass: HomeAssistant, tracker: 'DeviceTracker', discovery_info: Optional[Dict[str, Any]] = None) -> None:
        """Set up a legacy platform."""
        assert self.type == PLATFORM_TYPE_LEGACY
        full_name = f'{self.name}.{DOMAIN}'
        LOGGER.info('Setting up %s', full_name)
        with async_start_setup(hass, integration=self.name, group=str(id(self.config)), phase=SetupPhases.PLATFORM_SETUP):
            try:
                scanner = None
                setup = None
                if hasattr(self.platform, 'async_get_scanner'):
                    scanner = await self.platform.async_get_scanner(hass, {DOMAIN: self.config})
                elif hasattr(self.platform, 'get_scanner'):
                    scanner = await hass.async_add_executor_job(self.platform.get_scanner, hass, {DOMAIN: self.config})
                elif hasattr(self.platform, 'async_setup_scanner'):
                    setup = await self.platform.async_setup_scanner(hass, self.config, tracker.async_see, discovery_info)
                elif hasattr(self.platform, 'setup_scanner'):
                    setup = await hass.async_add_executor_job(self.platform.setup_scanner, hass, self.config, tracker.see, discovery_info)
                else:
                    raise HomeAssistantError('Invalid legacy device_tracker platform.')
                if scanner is not None:
                    async_setup_scanner_platform(hass, self.config, scanner, tracker.async_see, self.type)
                if not setup and scanner is None:
                    LOGGER.error('Error setting up platform %s %s', self.type, self.name)
                    return
                hass.config.components.add(full_name)
            except Exception:
                LOGGER.exception('Error setting up platform %s %s', self.type, self.name)

async def async_extract_config(hass: HomeAssistant, config: ConfigType) -> List['DeviceTrackerPlatform']:
    """Extract device tracker config and split between legacy and modern."""
    legacy: List[DeviceTrackerPlatform] = []
    for platform in await asyncio.gather(*(async_create_platform_type(hass, config, p_type, p_config) for p_type, p_config in config_per_platform(config, DOMAIN) if p_type is not None)):
        if platform is None:
            continue
        if platform.type == PLATFORM_TYPE_LEGACY:
            legacy.append(platform)
        else:
            raise ValueError(f'Unable to determine type for {platform.name}: {platform.type}')
    return legacy

async def async_create_platform_type(hass: HomeAssistant, config: ConfigType, p_type: str, p_config: ConfigType) -> Optional['DeviceTrackerPlatform']:
    """Determine type of platform."""
    platform = await async_prepare_setup_platform(hass, config, DOMAIN, p_type)
    if platform is None:
        return None
    return DeviceTrackerPlatform(p_type, platform, p_config)

def _load_device_names_and_attributes(scanner: 'DeviceScanner', device_name_uses_executor: bool, extra_attributes_uses_executor: bool, seen: Set[str], found_devices: List[str]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Load device names and attributes in a single executor job."""
    host_name_by_mac: Dict[str, str] = {}
    extra_attributes_by_mac: Dict[str, Dict[str, Any]] = {}
    for mac in found_devices:
        if device_name_uses_executor and mac not in seen:
            host_name_by_mac[mac] = scanner.get_device_name(mac)
        if extra_attributes_uses_executor:
            try:
                extra_attributes_by_mac[mac] = scanner.get_extra_attributes(mac)
            except NotImplementedError:
                extra_attributes_by_mac[mac] = {}
    return (host_name_by_mac, extra_attributes_by_mac)

@callback
def async_setup_scanner_platform(hass: HomeAssistant, config: ConfigType, scanner: 'DeviceScanner', async_see_device: AsyncSeeCallback, platform: str) -> None:
    """Set up the connect scanner-based platform to device tracker."""
    interval: timedelta = config.get(CONF_SCAN_INTERVAL, SCAN_INTERVAL)
    update_lock = asyncio.Lock()
    scanner.hass = hass
    seen: Set[str] = set()

    async def async_device_tracker_scan(now: datetime) -> None:
        """Handle interval matches."""
        if update_lock.locked():
            LOGGER.warning('Updating device list from %s took longer than the scheduled scan interval %s', platform, interval)
            return
        async with update_lock:
            found_devices = await scanner.async_scan_devices()
        device_name_uses_executor = scanner.async_get_device_name.__func__ is DeviceScanner.async_get_device_name
        extra_attributes_uses_executor = scanner.async_get_extra_attributes.__func__ is DeviceScanner.async_get_extra_attributes
        host_name_by_mac: Dict[str, str] = {}
        extra_attributes_by_mac: Dict[str, Dict[str, Any]] = {}
        if device_name_uses_executor or extra_attributes_uses_executor:
            host_name_by_mac, extra_attributes_by_mac = await hass.async_add_executor_job(_load_device_names_and_attributes, scanner, device_name_uses_executor, extra_attributes_uses_executor, seen, found_devices)
        for mac in found_devices:
            if mac in seen:
                host_name = None
            else:
                host_name = host_name_by_mac.get(mac, await scanner.async_get_device_name(mac))
                seen.add(mac)
            try:
                extra_attributes = extra_attributes_by_mac.get(mac, await scanner.async_get_extra_attributes(mac))
            except NotImplementedError:
                extra_attributes = {}
            kwargs = {'mac': mac, 'host_name': host_name, 'source_type': SourceType.ROUTER, 'attributes': {'scanner': scanner.__class__.__name__, **extra_attributes}}
            zone_home = hass.states.get(ENTITY_ID_HOME)
            if zone_home is not None:
                kwargs['gps'] = [zone_home.attributes[ATTR_LATITUDE], zone_home.attributes[ATTR_LONGITUDE]]
                kwargs['gps_accuracy'] = 0
            hass.async_create_task(async_see_device(**kwargs), eager_start=True)
    cancel_legacy_scan = async_track_time_interval(hass, async_device_tracker_scan, interval, name=f'device_tracker {platform} legacy scan')
    hass.async_create_task(async_device_tracker_scan(None), eager_start=True)

    @callback
    def _on_hass_stop(_: Event) -> None:
        """Cleanup when Home Assistant stops."""
        cancel_legacy_scan()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _on_hass_stop)

async def get_tracker(hass: HomeAssistant, config: ConfigType) -> 'DeviceTracker':
    """Create a tracker."""
    yaml_path = hass.config.path(YAML_DEVICES)
    conf = config.get(DOMAIN, [])
    conf = conf[0] if conf else {}
    consider_home = conf.get(CONF_CONSIDER_HOME, DEFAULT_CONSIDER_HOME)
    defaults = conf.get(CONF_NEW_DEVICE_DEFAULTS, {})
    if (track_new := conf.get(CONF_TRACK_NEW)) is None:
        track_new = defaults.get(CONF_TRACK_NEW, DEFAULT_TRACK_NEW)
    devices = await async_load_config(yaml_path, hass, consider_home)
    return DeviceTracker(hass, consider_home, track_new, defaults, devices)

class DeviceTracker:
    """Representation of a device tracker."""

    def __init__(self, hass: HomeAssistant, consider_home: timedelta, track_new: bool, defaults: Dict[str, Any], devices: List['Device']) -> None:
        """Initialize a device tracker."""
        self.hass = hass
        self.devices: Dict[str, 'Device'] = {dev.dev_id: dev for dev in devices}
        self.mac_to_dev: Dict[str, 'Device'] = {dev.mac: dev for dev in devices if dev.mac}
        self.consider_home = consider_home
        self.track_new = track_new if track_new is not None else defaults.get(CONF_TRACK_NEW, DEFAULT_TRACK_NEW)
        self.defaults = defaults
        self._is_updating = asyncio.Lock()
        for dev in devices:
            if self.devices[dev.dev_id] is not dev:
