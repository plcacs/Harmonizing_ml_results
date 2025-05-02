"""Support for Apple HomeKit."""
from __future__ import annotations
import asyncio
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
import ipaddress
import logging
import os
import socket
from typing import Any, cast, Optional, Dict, List, Set, Tuple, Union, Callable
from aiohttp import web
from pyhap import util as pyhap_util
from pyhap.characteristic import Characteristic
from pyhap.const import STANDALONE_AID
from pyhap.loader import get_loader
from pyhap.service import Service
import voluptuous as vol
from zeroconf.asyncio import AsyncZeroconf
from homeassistant.components import device_automation, network, zeroconf
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN, BinarySensorDeviceClass
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.device_automation.trigger import async_validate_trigger_config
from homeassistant.components.event import DOMAIN as EVENT_DOMAIN, EventDeviceClass
from homeassistant.components.http import KEY_HASS, HomeAssistantView
from homeassistant.components.humidifier import DOMAIN as HUMIDIFIER_DOMAIN
from homeassistant.components.lock import DOMAIN as LOCK_DOMAIN
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import ATTR_BATTERY_CHARGING, ATTR_BATTERY_LEVEL, ATTR_DEVICE_ID, ATTR_ENTITY_ID, ATTR_HW_VERSION, ATTR_MANUFACTURER, ATTR_MODEL, ATTR_SW_VERSION, CONF_DEVICES, CONF_IP_ADDRESS, CONF_NAME, CONF_PORT, EVENT_HOMEASSISTANT_STOP, SERVICE_RELOAD
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceCall, State, callback
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import config_validation as cv, device_registry as dr, entity_registry as er, instance_id
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entityfilter import BASE_FILTER_SCHEMA, FILTER_SCHEMA, EntityFilter
from homeassistant.helpers.reload import async_integration_yaml_config
from homeassistant.helpers.service import async_extract_referenced_entity_ids, async_register_admin_service
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import IntegrationNotFound, async_get_integration
from homeassistant.util.async_ import create_eager_task
from . import type_cameras, type_covers, type_fans, type_humidifiers, type_lights, type_locks, type_media_players, type_remotes, type_security_systems, type_sensors, type_switches, type_thermostats
from .accessories import HomeAccessory, HomeBridge, HomeDriver, get_accessory
from .aidmanager import AccessoryAidStorage
from .const import ATTR_INTEGRATION, BRIDGE_NAME, BRIDGE_SERIAL_NUMBER, CONF_ADVERTISE_IP, CONF_ENTITY_CONFIG, CONF_ENTRY_INDEX, CONF_EXCLUDE_ACCESSORY_MODE, CONF_FILTER, CONF_HOMEKIT_MODE, CONF_LINKED_BATTERY_CHARGING_SENSOR, CONF_LINKED_BATTERY_SENSOR, CONF_LINKED_DOORBELL_SENSOR, CONF_LINKED_HUMIDITY_SENSOR, CONF_LINKED_MOTION_SENSOR, CONFIG_OPTIONS, DEFAULT_EXCLUDE_ACCESSORY_MODE, DEFAULT_HOMEKIT_MODE, DEFAULT_PORT, DOMAIN, HOMEKIT_MODE_ACCESSORY, HOMEKIT_MODES, MANUFACTURER, PERSIST_LOCK_DATA, SERVICE_HOMEKIT_RESET_ACCESSORY, SERVICE_HOMEKIT_UNPAIR, SHUTDOWN_TIMEOUT, SIGNAL_RELOAD_ENTITIES
from .iidmanager import AccessoryIIDStorage
from .models import HomeKitConfigEntry, HomeKitEntryData
from .type_triggers import DeviceTriggerAccessory
from .util import accessory_friendly_name, async_dismiss_setup_message, async_port_is_available, async_show_setup_message, get_persist_fullpath_for_entry_id, remove_state_files_for_entry_id, state_needs_accessory_mode, validate_entity_config

_LOGGER: logging.Logger = logging.getLogger(__name__)
MAX_DEVICES: int = 150
STATUS_READY: int = 0
STATUS_RUNNING: int = 1
STATUS_STOPPED: int = 2
STATUS_WAIT: int = 3
PORT_CLEANUP_CHECK_INTERVAL_SECS: int = 1
_HOMEKIT_CONFIG_UPDATE_TIME: int = 10
_HAS_IPV6: bool = hasattr(socket, 'AF_INET6')
_DEFAULT_BIND: List[str] = ['0.0.0.0', '::'] if _HAS_IPV6 else ['0.0.0.0']
BATTERY_CHARGING_SENSOR: Tuple[str, BinarySensorDeviceClass] = (BINARY_SENSOR_DOMAIN, BinarySensorDeviceClass.BATTERY_CHARGING)
BATTERY_SENSOR: Tuple[str, SensorDeviceClass] = (SENSOR_DOMAIN, SensorDeviceClass.BATTERY)
MOTION_EVENT_SENSOR: Tuple[str, EventDeviceClass] = (EVENT_DOMAIN, EventDeviceClass.MOTION)
MOTION_SENSOR: Tuple[str, BinarySensorDeviceClass] = (BINARY_SENSOR_DOMAIN, BinarySensorDeviceClass.MOTION)
DOORBELL_EVENT_SENSOR: Tuple[str, EventDeviceClass] = (EVENT_DOMAIN, EventDeviceClass.DOORBELL)
HUMIDITY_SENSOR: Tuple[str, SensorDeviceClass] = (SENSOR_DOMAIN, SensorDeviceClass.HUMIDITY)

def _has_all_unique_names_and_ports(bridges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate that each homekit bridge configured has a unique name."""
    names: List[str] = [bridge[CONF_NAME] for bridge in bridges]
    ports: List[int] = [bridge[CONF_PORT] for bridge in bridges]
    vol.Schema(vol.Unique())(names)
    vol.Schema(vol.Unique())(ports)
    return bridges

BRIDGE_SCHEMA: vol.Schema = vol.All(
    vol.Schema({
        vol.Optional(CONF_HOMEKIT_MODE, default=DEFAULT_HOMEKIT_MODE): vol.In(HOMEKIT_MODES),
        vol.Optional(CONF_NAME, default=BRIDGE_NAME): vol.All(cv.string, vol.Length(min=3, max=25)),
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
        vol.Optional(CONF_IP_ADDRESS): vol.All(ipaddress.ip_address, cv.string),
        vol.Optional(CONF_ADVERTISE_IP): vol.All(cv.ensure_list, [ipaddress.ip_address], [cv.string]),
        vol.Optional(CONF_FILTER, default={}): BASE_FILTER_SCHEMA,
        vol.Optional(CONF_ENTITY_CONFIG, default={}): validate_entity_config,
        vol.Optional(CONF_DEVICES): cv.ensure_list,
    }, extra=vol.ALLOW_EXTRA)
)

CONFIG_SCHEMA: vol.Schema = vol.Schema(
    {DOMAIN: vol.All(cv.ensure_list, [BRIDGE_SCHEMA], _has_all_unique_names_and_ports)},
    extra=vol.ALLOW_EXTRA
)

RESET_ACCESSORY_SERVICE_SCHEMA: vol.Schema = vol.Schema(
    {vol.Required(ATTR_ENTITY_ID): cv.entity_ids}
)

UNPAIR_SERVICE_SCHEMA: vol.Schema = vol.All(
    vol.Schema(cv.ENTITY_SERVICE_FIELDS),
    cv.has_at_least_one_key(ATTR_DEVICE_ID)
)

@callback
def _async_all_homekit_instances(hass: HomeAssistant) -> List[Any]:
    """All active HomeKit instances."""
    return [hk_data.homekit for entry in hass.config_entries.async_entries(DOMAIN) if (hk_data := getattr(entry, 'runtime_data', None))]

@callback
def _async_get_imported_entries_indices(current_entries: List[ConfigEntry]) -> Tuple[Dict[str, ConfigEntry], Dict[int, ConfigEntry]]:
    """Return a dicts of the entries by name and port."""
    entries_by_name: Dict[str, ConfigEntry] = {}
    entries_by_port: Dict[int, ConfigEntry] = {}
    for entry in current_entries:
        if entry.source != SOURCE_IMPORT:
            continue
        entries_by_name[entry.data.get(CONF_NAME, BRIDGE_NAME)] = entry
        entries_by_port[entry.data.get(CONF_PORT, DEFAULT_PORT)] = entry
    return (entries_by_name, entries_by_port)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the HomeKit from yaml."""
    hass.data[PERSIST_LOCK_DATA] = asyncio.Lock()
    await hass.async_add_executor_job(get_loader)
    _async_register_events_and_services(hass)
    if DOMAIN not in config:
        return True
    current_entries: List[ConfigEntry] = hass.config_entries.async_entries(DOMAIN)
    entries_by_name, entries_by_port = _async_get_imported_entries_indices(current_entries)
    for index, conf in enumerate(config[DOMAIN]):
        if _async_update_config_entry_from_yaml(hass, entries_by_name, entries_by_port, conf):
            continue
        conf[CONF_ENTRY_INDEX] = index
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN, context={'source': SOURCE_IMPORT}, data=conf
            ), eager_start=True
        )
    return True

@callback
def _async_update_config_entry_from_yaml(
    hass: HomeAssistant,
    entries_by_name: Dict[str, ConfigEntry],
    entries_by_port: Dict[int, ConfigEntry],
    conf: Dict[str, Any]
) -> bool:
    """Update a config entry with the latest yaml."""
    if not (matching_entry := (entries_by_name.get(conf.get(CONF_NAME, BRIDGE_NAME)) or entries_by_port.get(conf.get(CONF_PORT, DEFAULT_PORT)))):
        return False
    data = conf.copy()
    options: Dict[str, Any] = {}
    for key in CONFIG_OPTIONS:
        if key in data:
            options[key] = data[key]
            del data[key]
    hass.config_entries.async_update_entry(matching_entry, data=data, options=options)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HomeKit from a config entry."""
    _async_import_options_from_data_if_missing(hass, entry)
    conf: Dict[str, Any] = entry.data
    options: Dict[str, Any] = entry.options
    name: str = conf[CONF_NAME]
    port: int = conf[CONF_PORT]
    _LOGGER.debug('Begin setup HomeKit for %s', name)
    ip_address: Union[str, List[str]] = conf.get(CONF_IP_ADDRESS, _DEFAULT_BIND)
    advertise_ips: List[str] = conf.get(CONF_ADVERTISE_IP) or await network.async_get_announce_addresses(hass)
    exclude_accessory_mode: bool = conf.get(CONF_EXCLUDE_ACCESSORY_MODE, DEFAULT_EXCLUDE_ACCESSORY_MODE)
    homekit_mode: str = options.get(CONF_HOMEKIT_MODE, DEFAULT_HOMEKIT_MODE)
    entity_config: Dict[str, Any] = options.get(CONF_ENTITY_CONFIG, {}).copy()
    entity_filter: EntityFilter = FILTER_SCHEMA(options.get(CONF_FILTER, {}))
    devices: List[str] = options.get(CONF_DEVICES, [])
    homekit = HomeKit(
        hass, name, port, ip_address, entity_filter, exclude_accessory_mode,
        entity_config, homekit_mode, advertise_ips, entry.entry_id, entry.title,
        devices=devices
    )
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    entry.async_on_unload(hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, homekit.async_stop))
    entry_data = HomeKitEntryData(homekit=homekit, pairing_qr=None, pairing_qr_secret=None)
    entry.runtime_data = entry_data

    async def _async_start_homekit(hass: HomeAssistant) -> None:
        await homekit.async_start()
    entry.async_on_unload(async_at_started(hass, _async_start_homekit))
    return True

async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    if entry.source == SOURCE_IMPORT:
        return
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    async_dismiss_setup_message(hass, entry.entry_id)
    entry_data: HomeKitEntryData = entry.runtime_data
    homekit: HomeKit = entry_data.homekit
    if homekit.status == STATUS_RUNNING:
        await homekit.async_stop()
    logged_shutdown_wait: bool = False
    for _ in range(SHUTDOWN_TIMEOUT):
        if async_port_is_available(entry.data[CONF_PORT]):
            break
        if not logged_shutdown_wait:
            _LOGGER.debug('Waiting for the HomeKit server to shutdown')
            logged_shutdown_wait = True
        await asyncio.sleep(PORT_CLEANUP_CHECK_INTERVAL_SECS)
    return True

async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Remove a config entry."""
    await hass.async_add_executor_job(remove_state_files_for_entry_id, hass, entry.entry_id)

@callback
def _async_import_options_from_data_if_missing(hass: HomeAssistant, entry: ConfigEntry) -> None:
    options = deepcopy(dict(entry.options))
    data = deepcopy(dict(entry.data))
    modified: bool = False
    for importable_option in CONFIG_OPTIONS:
        if importable_option not in entry.options and importable_option in entry.data:
            options[importable_option] = entry.data[importable_option]
            del data[importable_option]
            modified = True
    if modified:
        hass.config_entries.async_update_entry(entry, data=data, options=options)

@callback
def _async_register_events_and_services(hass: HomeAssistant) -> None:
    """Register events and services for HomeKit."""
    hass.http.register_view(HomeKitPairingQRView)

    async def async_handle_homekit_reset_accessory(service: ServiceCall) -> None:
        """Handle reset accessory HomeKit service call."""
        for homekit in _async_all_homekit_instances(hass):
            if homekit.status != STATUS_RUNNING:
                _LOGGER.warning('HomeKit is not running. Either it is waiting to be started or has been stopped')
                continue
            entity_ids: List[str] = cast(List[str], service.data.get('entity_id'))
            await homekit.async_reset_accessories(entity_ids)
    hass.services.async_register(
        DOMAIN, SERVICE_HOMEKIT_RESET_ACCESSORY,
        async_handle_homekit_reset_accessory,
        schema=RESET_ACCESSORY_SERVICE_SCHEMA
    )

    async def async_handle_homekit_unpair(service: ServiceCall) -> None:
        """Handle unpair HomeKit service call."""
        referenced = async_extract_referenced_entity_ids(hass, service)
        dev_reg = dr.async_get(hass)
        for device_id in referenced.referenced_devices:
            if not (dev_reg_ent := dev_reg.async_get(device_id)):
                raise HomeAssistantError(f'No device found for device id: {device_id}')
            macs = [cval for ctype, cval in dev_reg_ent.connections if ctype == dr.CONNECTION_NETWORK_MAC]
            matching_instances = [
                homekit for homekit in _async_all_homekit_instances(hass)
                if homekit.driver and dr.format_mac(homekit.driver.state.mac) in macs
            ]
            if not matching_instances:
                raise HomeAssistantError(f'No homekit accessory found for device id: {device_id}')
            for homekit in matching_instances:
                homekit.async_unpair()
    hass.services.async_register(
        DOMAIN, SERVICE_HOMEKIT_UNPAIR,
        async_handle_homekit_unpair,
        schema=UNPAIR_SERVICE_SCHEMA
    )

    async def _handle_homekit_reload(service: ServiceCall) -> None:
        """Handle start HomeKit service call."""
        config = await async_integration_yaml_config(hass, DOMAIN)
        if not config or DOMAIN not in config:
            return
        current_entries = hass.config_entries.async_entries(DOMAIN)
        entries_by_name, entries_by_port = _async_get_imported_entries_indices(current_entries)
        for conf in config[DOMAIN]:
            _async_update_config_entry_from_yaml(hass, entries_by_name, entries_by_port, conf)
        reload_tasks = [
            create_eager_task(hass