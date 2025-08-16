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
_LOGGER: logging.Logger
MAX_DEVICES: int
STATUS_READY: int
STATUS_RUNNING: int
STATUS_STOPPED: int
STATUS_WAIT: int
PORT_CLEANUP_CHECK_INTERVAL_SECS: int
_HOMEKIT_CONFIG_UPDATE_TIME: int
_HAS_IPV6: bool
_DEFAULT_BIND: list[str]
BATTERY_CHARGING_SENSOR: tuple[str, str]
BATTERY_SENSOR: tuple[str, str]
MOTION_EVENT_SENSOR: tuple[str, str]
MOTION_SENSOR: tuple[str, str]
DOORBELL_EVENT_SENSOR: tuple[str, str]
HUMIDITY_SENSOR: tuple[str, str]

def _has_all_unique_names_and_ports(bridges: list[dict[str, Any]]) -> list[dict[str, Any]]:
BRIDGE_SCHEMA: vol.Schema
CONFIG_SCHEMA: vol.Schema
RESET_ACCESSORY_SERVICE_SCHEMA: vol.Schema
UNPAIR_SERVICE_SCHEMA: vol.Schema

def _async_all_homekit_instances(hass: HomeAssistant) -> list[HomeKit]:
def _async_get_imported_entries_indices(current_entries: list[ConfigEntry]) -> tuple[dict[str, ConfigEntry], dict[int, ConfigEntry]]:

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:

async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry):

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:

async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry):

@callback
def _async_import_options_from_data_if_missing(hass: HomeAssistant, entry: ConfigEntry):

@callback
def _async_register_events_and_services(hass: HomeAssistant):

class HomeKit:

    def __init__(self, hass: HomeAssistant, name: str, port: int, ip_address: str, entity_filter: EntityFilter, exclude_accessory_mode: bool, entity_config: dict[str, Any], homekit_mode: str, advertise_ips: list[str], entry_id: str, entry_title: str, devices: list[str] = None):

    def setup(self, async_zeroconf_instance: AsyncZeroconf, uuid: str) -> bool:

    async def async_reset_accessories(self, entity_ids: list[str]):

    async def async_reload_accessories(self, entity_ids: list[str]):

    @callback
    def _async_shutdown_accessory(self, accessory: HomeAccessory):

    async def _async_reload_accessories_in_accessory_mode(self, entity_ids: list[str]):

    def _async_remove_accessories_by_entity_id(self, entity_ids: list[str]) -> list[str]:

    async def _async_reset_accessories_in_bridge_mode(self, entity_ids: list[str]):

    async def _async_reload_accessories_in_bridge_mode(self, entity_ids: list[str]):

    async def _async_recreate_removed_accessories_in_bridge_mode(self, removed: list[str]):

    @callback
    def _async_update_accessories_hash(self) -> bool:

    def add_bridge_accessory(self, state: State) -> HomeAccessory:

    def _would_exceed_max_devices(self, name: str) -> bool:

    async def add_bridge_triggers_accessory(self, device: Device, device_triggers: list[dict[str, Any]]):

    @callback
    def async_remove_bridge_accessory(self, aid: int) -> HomeAccessory:

    async def async_configure_accessories(self) -> list[State]:

    async def async_start(self, *args):

    @callback
    def _async_show_setup_message(self):

    @callback
    def async_unpair(self):

    @callback
    def _async_register_bridge(self):

    @callback
    def _async_purge_old_bridges(self, dev_reg: DeviceRegistry, identifier: tuple[str, str, str], connection: tuple[str, str]):

    @callback
    def _async_create_single_accessory(self, entity_states: list[State]) -> HomeAccessory:

    async def _async_create_bridge_accessory(self, entity_states: list[State]) -> HomeBridge:

    async def _async_add_trigger_accessories(self):

    async def _async_create_accessories(self) -> bool:

    async def async_stop(self, *args):

    @callback
    def _async_configure_linked_sensors(self, ent_reg_ent: RegistryEntry, lookup: dict[tuple[str, str], str], state: State):

    async def _async_set_device_info_attributes(self, ent_reg_ent: RegistryEntry, dev_reg: DeviceRegistry, entity_id: str):

    def _fill_config_from_device_registry_entry(self, device_entry: DeviceEntry, config: dict[str, Any]):

class HomeKitPairingQRView(HomeAssistantView):

    async def get(self, request: web.Request) -> web.Response:
