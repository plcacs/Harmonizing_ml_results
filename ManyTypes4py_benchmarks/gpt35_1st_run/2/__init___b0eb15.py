from __future__ import annotations
import asyncio
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
import ipaddress
import logging
import os
import socket
from typing import Any, cast
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
_LOGGER = logging.getLogger(__name__)
MAX_DEVICES = 150
STATUS_READY = 0
STATUS_RUNNING = 1
STATUS_STOPPED = 2
STATUS_WAIT = 3
PORT_CLEANUP_CHECK_INTERVAL_SECS = 1
_HOMEKIT_CONFIG_UPDATE_TIME = 10
_HAS_IPV6 = hasattr(socket, 'AF_INET6')
_DEFAULT_BIND = ['0.0.0.0', '::'] if _HAS_IPV6 else ['0.0.0.0']
BATTERY_CHARGING_SENSOR = (BINARY_SENSOR_DOMAIN, BinarySensorDeviceClass.BATTERY_CHARGING)
BATTERY_SENSOR = (SENSOR_DOMAIN, SensorDeviceClass.BATTERY)
MOTION_EVENT_SENSOR = (EVENT_DOMAIN, EventDeviceClass.MOTION)
MOTION_SENSOR = (BINARY_SENSOR_DOMAIN, BinarySensorDeviceClass.MOTION)
DOORBELL_EVENT_SENSOR = (EVENT_DOMAIN, EventDeviceClass.DOORBELL)
HUMIDITY_SENSOR = (SENSOR_DOMAIN, SensorDeviceClass.HUMIDITY)

def _has_all_unique_names_and_ports(bridges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate that each homekit bridge configured has a unique name."""
    names = [bridge[CONF_NAME] for bridge in bridges]
    ports = [bridge[CONF_PORT] for bridge in bridges]
    vol.Schema(vol.Unique())(names)
    vol.Schema(vol.Unique())(ports)
    return bridges

BRIDGE_SCHEMA = vol.All(
    vol.Schema({
        vol.Optional(CONF_HOMEKIT_MODE, default=DEFAULT_HOMEKIT_MODE): vol.In(HOMEKIT_MODES),
        vol.Optional(CONF_NAME, default=BRIDGE_NAME): vol.All(cv.string, vol.Length(min=3, max=25)),
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
        vol.Optional(CONF_IP_ADDRESS): vol.All(ipaddress.ip_address, cv.string),
        vol.Optional(CONF_ADVERTISE_IP): vol.All(cv.ensure_list, [ipaddress.ip_address], [cv.string]),
        vol.Optional(CONF_FILTER, default={}): BASE_FILTER_SCHEMA,
        vol.Optional(CONF_ENTITY_CONFIG, default={}): validate_entity_config,
        vol.Optional(CONF_DEVICES): cv.ensure_list
    }, extra=vol.ALLOW_EXTRA)
)

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.All(cv.ensure_list, [BRIDGE_SCHEMA], _has_all_unique_names_and_ports)
}, extra=vol.ALLOW_EXTRA)

RESET_ACCESSORY_SERVICE_SCHEMA = vol.Schema({
    vol.Required(ATTR_ENTITY_ID): cv.entity_ids
})

UNPAIR_SERVICE_SCHEMA = vol.All(
    vol.Schema(cv.ENTITY_SERVICE_FIELDS),
    cv.has_at_least_one_key(ATTR_DEVICE_ID)
)

def _async_all_homekit_instances(hass: HomeAssistant) -> list[HomeKit]:
    """All active HomeKit instances."""
    return [hk_data.homekit for entry in hass.config_entries.async_entries(DOMAIN) if (hk_data := getattr(entry, 'runtime_data', None))]

def _async_get_imported_entries_indices(current_entries: list[ConfigEntry]) -> tuple[dict[str, ConfigEntry], dict[int, ConfigEntry]]:
    """Return a dicts of the entries by name and port."""
    entries_by_name = {}
    entries_by_port = {}
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
    current_entries = hass.config_entries.async_entries(DOMAIN)
    entries_by_name, entries_by_port = _async_get_imported_entries_indices(current_entries)
    for index, conf in enumerate(config[DOMAIN]):
        if _async_update_config_entry_from_yaml(hass, entries_by_name, entries_by_port, conf):
            continue
        conf[CONF_ENTRY_INDEX] = index
        hass.async_create_task(hass.config_entries.flow.async_init(DOMAIN, context={'source': SOURCE_IMPORT}, data=conf), eager_start=True)
    return True

@callback
def _async_update_config_entry_from_yaml(hass: HomeAssistant, entries_by_name: dict[str, ConfigEntry], entries_by_port: dict[int, ConfigEntry], conf: dict[str, Any]) -> bool:
    """Update a config entry with the latest yaml.

    Returns True if a matching config entry was found

    Returns False if there is no matching config entry
    """
    if not (matching_entry := (entries_by_name.get(conf.get(CONF_NAME, BRIDGE_NAME)) or entries_by_port.get(conf.get(CONF_PORT, DEFAULT_PORT)))):
        return False
    data = conf.copy()
    options = {}
    for key in CONFIG_OPTIONS:
        if key in data:
            options[key] = data[key]
            del data[key]
    hass.config_entries.async_update_entry(matching_entry, data=data, options=options)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HomeKit from a config entry."""
    _async_import_options_from_data_if_missing(hass, entry)
    conf = entry.data
    options = entry.options
    name = conf[CONF_NAME]
    port = conf[CONF_PORT]
    _LOGGER.debug('Begin setup HomeKit for %s', name)
    ip_address = conf.get(CONF_IP_ADDRESS, _DEFAULT_BIND)
    advertise_ips = conf.get(CONF_ADVERTISE_IP) or await network.async_get_announce_addresses(hass)
    exclude_accessory_mode = conf.get(CONF_EXCLUDE_ACCESSORY_MODE, DEFAULT_EXCLUDE_ACCESSORY_MODE)
    homekit_mode = options.get(CONF_HOMEKIT_MODE, DEFAULT_HOMEKIT_MODE)
    entity_config = options.get(CONF_ENTITY_CONFIG, {}).copy()
    entity_filter = FILTER_SCHEMA(options.get(CONF_FILTER, {}))
    devices = options.get(CONF_DEVICES, [])
    homekit = HomeKit(hass, name, port, ip_address, entity_filter, exclude_accessory_mode, entity_config, homekit_mode, advertise_ips, entry.entry_id, entry.title, devices=devices)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    entry.async_on_unload(hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, homekit.async_stop))
    entry_data = HomeKitEntryData(homekit=homekit, pairing_qr=None, pairing_qr_secret=None)
    entry.runtime_data = entry_data

    async def _async_start_homekit(hass: HomeAssistant):
        await homekit.async_start()
    entry.async_on_unload(async_at_started(hass, _async_start_homekit))
    return True

async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    if entry.source == SOURCE_IMPORT:
        return
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    async_dismiss_setup_message(hass, entry.entry_id)
    entry_data = entry.runtime_data
    homekit = entry_data.homekit
    if homekit.status == STATUS_RUNNING:
        await homekit.async_stop()
    logged_shutdown_wait = False
    for _ in range(SHUTDOWN_TIMEOUT):
        if async_port_is_available(entry.data[CONF_PORT]):
            break
        if not logged_shutdown_wait:
            _LOGGER.debug('Waiting for the HomeKit server to shutdown')
            logged_shutdown_wait = True
        await asyncio.sleep(PORT_CLEANUP_CHECK_INTERVAL_SECS)
    return True

async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Remove a config entry."""
    await hass.async_add_executor_job(remove_state_files_for_entry_id, hass, entry.entry_id)

@callback
def _async_import_options_from_data_if_missing(hass: HomeAssistant, entry: ConfigEntry):
    options = deepcopy(dict(entry.options))
    data = deepcopy(dict(entry.data))
    modified = False
    for importable_option in CONFIG_OPTIONS:
        if importable_option not in entry.options and importable_option in entry.data:
            options[importable_option] = entry.data[importable_option]
            del data[importable_option]
            modified = True
    if modified:
        hass.config_entries.async_update_entry(entry, data=data, options=options)

@callback
def _async_register_events_and_services(hass: HomeAssistant):
    """Register events and services for HomeKit."""
    hass.http.register_view(HomeKitPairingQRView)

    async def async_handle_homekit_reset_accessory(service: ServiceCall):
        """Handle reset accessory HomeKit service call."""
        for homekit in _async_all_homekit_instances(hass):
            if homekit.status != STATUS_RUNNING:
                _LOGGER.warning('HomeKit is not running. Either it is waiting to be started or has been stopped')
                continue
            entity_ids = cast(list[str], service.data.get('entity_id'))
            await homekit.async_reset_accessories(entity_ids)
    hass.services.async_register(DOMAIN, SERVICE_HOMEKIT_RESET_ACCESSORY, async_handle_homekit_reset_accessory, schema=RESET_ACCESSORY_SERVICE_SCHEMA)

    async def async_handle_homekit_unpair(service: ServiceCall):
        """Handle unpair HomeKit service call."""
        referenced = async_extract_referenced_entity_ids(hass, service)
        dev_reg = dr.async_get(hass)
        for device_id in referenced.referenced_devices:
            if not (dev_reg_ent := dev_reg.async_get(device_id)):
                raise HomeAssistantError(f'No device found for device id: {device_id}')
            macs = [cval for ctype, cval in dev_reg_ent.connections if ctype == dr.CONNECTION_NETWORK_MAC]
            matching_instances = [homekit for homekit in _async_all_homekit_instances(hass) if homekit.driver and dr.format_mac(homekit.driver.state.mac) in macs]
            if not matching_instances:
                raise HomeAssistantError(f'No homekit accessory found for device id: {device_id}')
            for homekit in matching_instances:
                homekit.async_unpair()
    hass.services.async_register(DOMAIN, SERVICE_HOMEKIT_UNPAIR, async_handle_homekit_unpair, schema=UNPAIR_SERVICE_SCHEMA)

    async def _handle_homekit_reload(service: ServiceCall):
        """Handle start HomeKit service call."""
        config = await async_integration_yaml_config(hass, DOMAIN)
        if not config or DOMAIN not in config:
            return
        current_entries = hass.config_entries.async_entries(DOMAIN)
        entries_by_name, entries_by_port = _async_get_imported_entries_indices(current_entries)
        for conf in config[DOMAIN]:
            _async_update_config_entry_from_yaml(hass, entries_by_name, entries_by_port, conf)
        reload_tasks = [create_eager_task(hass.config_entries.async_reload(entry.entry_id)) for entry in current_entries]
        await asyncio.gather(*reload_tasks)
    async_register_admin_service(hass, DOMAIN, SERVICE_RELOAD, _handle_homekit_reload)

class HomeKit:
    """Class to handle all actions between HomeKit and Home Assistant."""

    def __init__(self, hass: HomeAssistant, name: str, port: int, ip_address: str, entity_filter: EntityFilter, exclude_accessory_mode: bool, entity_config: dict[str, Any], homekit_mode: str, advertise_ips: list[str], entry_id: str, entry_title: str, devices: list[str] = None):
        """Initialize a HomeKit object."""
        self.hass = hass
        self._name = name
        self._port = port
        self._ip_address = ip_address
        self._filter = entity_filter
        self._config = defaultdict(dict, entity_config)
        self._exclude_accessory_mode = exclude_accessory_mode
        self._advertise_ips = advertise_ips
        self._entry_id = entry_id
        self._entry_title = entry_title
        self._homekit_mode = homekit_mode
        self._devices = devices or []
        self.aid_storage = None
        self.iid_storage = None
        self.status = STATUS_READY
        self.driver = None
        self.bridge = None
        self._reset_lock = asyncio.Lock()
        self._cancel_reload_dispatcher = None

    def setup(self, async_zeroconf_instance: AsyncZeroconf, uuid: str) -> bool:
        """Set up bridge and accessory driver.

        Returns True if data was loaded from disk

        Returns False if the persistent data was not loaded
        """
        assert self.iid_storage is not None
        persist_file = get_persist_fullpath_for_entry_id(self.hass, self._entry_id)
        self.driver = HomeDriver(self.hass, self._entry_id, self._name, self._entry_title, loop=self.hass.loop, address=self._ip_address, port=self._port, persist_file=persist_file, advertised_address=self._advertise_ips, async_zeroconf_instance=async_zeroconf_instance, zeroconf_server=f'{uuid}-hap.local.', loader=get_loader(), iid_storage=self.iid_storage)
        if os.path.exists(persist_file):
            self.driver.load()
            return True
        self.driver.state.mac = pyhap_util.generate_mac()
        return False

    async def async_reset_accessories(self, entity_ids: list[str]):
        """Reset the accessory to load the latest configuration."""
        _LOGGER.debug('Resetting accessories: %s', entity_ids)
        async with self._reset_lock:
            if not self.bridge:
                await self._async_reload_accessories_in_accessory_mode(entity_ids)
                return
            await self._async_reset_accessories_in_bridge_mode(entity_ids)

    async def async_reload_accessories(self, entity_ids: list[str]):
        """Reload the accessory to load the latest configuration."""
        _LOGGER.debug('Reloading accessories: %s', entity_ids)
        async with self._reset_lock:
            if not self.bridge:
                await self._async_reload_accessories_in_accessory_mode(entity_ids)
                return
            await self._async_reload_accessories_in_bridge_mode(entity_ids)

    @callback
    def _async_shutdown_accessory(self, accessory: HomeAccessory):
        """Shutdown an accessory."""
        assert self.driver is not None
        accessory.async_stop()
        iid_manager = accessory.iid_manager
        services = accessory.services
        for service in services:
            iid_manager.remove_obj(service)
            characteristics = service.characteristics
            for char in characteristics:
                iid_manager.remove_obj(char)

    async def _async_reload_accessories_in_accessory_mode(self, entity_ids: list[str]):
        """Reset accessories in accessory mode."""
        assert self.driver is not None
        acc = cast(HomeAccessory, self.driver.accessory)
        if acc.entity_id not in entity_ids:
            return
        if not (state := self.hass.states.get(acc.entity_id)):
            _LOGGER.warning('The underlying entity %s disappeared during reload', acc.entity_id)
            return
        self._async_shutdown_accessory(acc)
        if (new_acc := self._async_create_single_accessory([state])):
            self.driver.accessory = new_acc
            new_acc.run()
            self._async_update_accessories_hash()

    def _async_remove_accessories_by_entity_id(self, entity_ids: list[str]) -> list[str]:
        """Remove accessories by entity id."""
        assert self.aid_storage is not None
        assert self.bridge