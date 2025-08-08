from __future__ import annotations
import asyncio
import contextlib
import logging
import ssl
from typing import Any, cast, Dict, List, Union
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import ATTR_DEVICE_ID, CONF_HOST, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv, device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.typing import ConfigType
from .const import ACTION_PRESS, ACTION_RELEASE, ATTR_ACTION, ATTR_AREA_NAME, ATTR_BUTTON_NUMBER, ATTR_BUTTON_TYPE, ATTR_DEVICE_NAME, ATTR_LEAP_BUTTON_NUMBER, ATTR_SERIAL, ATTR_TYPE, BRIDGE_DEVICE_ID, BRIDGE_TIMEOUT, CONF_CA_CERTS, CONF_CERTFILE, CONF_KEYFILE, CONF_SUBTYPE, DOMAIN, LUTRON_CASETA_BUTTON_EVENT, MANUFACTURER, UNASSIGNED_AREA
from .device_trigger import DEVICE_TYPE_SUBTYPE_MAP_TO_LIP, KEYPAD_LEAP_BUTTON_NAME_OVERRIDE, LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP, LUTRON_BUTTON_TRIGGER_SCHEMA
from .models import LUTRON_BUTTON_LEAP_BUTTON_NUMBER, LUTRON_KEYPAD_AREA_NAME, LUTRON_KEYPAD_BUTTONS, LUTRON_KEYPAD_DEVICE_REGISTRY_DEVICE_ID, LUTRON_KEYPAD_LUTRON_DEVICE_ID, LUTRON_KEYPAD_MODEL, LUTRON_KEYPAD_NAME, LUTRON_KEYPAD_SERIAL, LUTRON_KEYPAD_TYPE, LutronButton, LutronCasetaConfigEntry, LutronCasetaData, LutronKeypad, LutronKeypadData
from .util import area_name_from_id, serial_to_unique_id

_LOGGER = logging.getLogger(__name__)
DATA_BRIDGE_CONFIG: str = 'lutron_caseta_bridges'
CONFIG_SCHEMA: vol.Schema = vol.Schema({DOMAIN: vol.All(cv.ensure_list, [{vol.Required(CONF_HOST): cv.string, vol.Required(CONF_KEYFILE): cv.string, vol.Required(CONF_CERTFILE): cv.string, vol.Required(CONF_CA_CERTS): cv.string}])}, extra=vol.ALLOW_EXTRA)
PLATFORMS: List[Platform] = [Platform.BINARY_SENSOR, Platform.BUTTON, Platform.COVER, Platform.FAN, Platform.LIGHT, Platform.SCENE, Platform.SWITCH]

async def async_setup(hass: HomeAssistant, base_config: ConfigType) -> bool:
    ...

async def _async_migrate_unique_ids(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> None:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    ...

@callback
def _async_register_bridge_device(hass: HomeAssistant, config_entry_id: str, bridge_device: Dict[str, Any], bridge: Smartbridge) -> None:
    ...

@callback
def _async_setup_keypads(hass: HomeAssistant, config_entry_id: str, bridge: Smartbridge, bridge_device: Dict[str, Any]) -> LutronKeypadData:
    ...

@callback
def _async_build_trigger_schemas(keypad_button_names_to_leap: Dict[int, Dict[str, Any]]) -> Dict[int, vol.Schema]:
    ...

@callback
def _async_build_lutron_keypad(bridge: Smartbridge, bridge_device: Dict[str, Any], bridge_keypad: Dict[str, Any], keypad_device_id: int) -> Dict[str, Union[str, int, List[int], DeviceInfo]]:
    ...

def _get_button_name(keypad: Dict[str, Any], bridge_button: Dict[str, Any]) -> str:
    ...

def _get_button_name_from_triggers(keypad: Dict[str, Any], button_number: int) -> str:
    ...

def _handle_none_keypad_serial(keypad_device: Dict[str, Any], bridge_serial: str) -> str:
    ...

@callback
def async_get_lip_button(device_type: str, leap_button: int) -> Union[int, None]:
    ...

@callback
def _async_subscribe_keypad_events(hass: HomeAssistant, bridge: Smartbridge, keypads: Dict[int, Dict[str, Any]], keypad_buttons: Dict[int, LutronButton], leap_to_keypad_button_names: Dict[int, Dict[int, str]]) -> None:
    ...

async def async_unload_entry(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    ...

def _id_to_identifier(lutron_id: str) -> Tuple[str, str]:
    ...

async def async_remove_config_entry_device(hass: HomeAssistant, entry: config_entries.ConfigEntry, device_entry: DeviceEntry) -> bool:
    ...
