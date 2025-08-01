"""Component for interacting with a Lutron Caseta system."""
from __future__ import annotations
import asyncio
import contextlib
from itertools import chain
import logging
import ssl
from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union, Callable
from pylutron_caseta import BUTTON_STATUS_PRESSED
from pylutron_caseta.smartbridge import Smartbridge
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import ATTR_DEVICE_ID, CONF_HOST, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv, device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo, DeviceEntry
from homeassistant.helpers.typing import ConfigType
from .const import ACTION_PRESS, ACTION_RELEASE, ATTR_ACTION, ATTR_AREA_NAME, ATTR_BUTTON_NUMBER, ATTR_BUTTON_TYPE, ATTR_DEVICE_NAME, ATTR_LEAP_BUTTON_NUMBER, ATTR_SERIAL, ATTR_TYPE, BRIDGE_DEVICE_ID, BRIDGE_TIMEOUT, CONF_CA_CERTS, CONF_CERTFILE, CONF_KEYFILE, CONF_SUBTYPE, DOMAIN, LUTRON_CASETA_BUTTON_EVENT, MANUFACTURER, UNASSIGNED_AREA
from .device_trigger import DEVICE_TYPE_SUBTYPE_MAP_TO_LIP, KEYPAD_LEAP_BUTTON_NAME_OVERRIDE, LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP, LUTRON_BUTTON_TRIGGER_SCHEMA
from .models import LUTRON_BUTTON_LEAP_BUTTON_NUMBER, LUTRON_KEYPAD_AREA_NAME, LUTRON_KEYPAD_BUTTONS, LUTRON_KEYPAD_DEVICE_REGISTRY_DEVICE_ID, LUTRON_KEYPAD_LUTRON_DEVICE_ID, LUTRON_KEYPAD_MODEL, LUTRON_KEYPAD_NAME, LUTRON_KEYPAD_SERIAL, LUTRON_KEYPAD_TYPE, LutronButton, LutronCasetaConfigEntry, LutronCasetaData, LutronKeypad, LutronKeypadData
from .util import area_name_from_id, serial_to_unique_id
_LOGGER = logging.getLogger(__name__)
DATA_BRIDGE_CONFIG = 'lutron_caseta_bridges'
CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.All(cv.ensure_list, [{vol.Required(CONF_HOST): cv.string, vol.Required(CONF_KEYFILE): cv.string, vol.Required(CONF_CERTFILE): cv.string, vol.Required(CONF_CA_CERTS): cv.string}])}, extra=vol.ALLOW_EXTRA)
PLATFORMS = [Platform.BINARY_SENSOR, Platform.BUTTON, Platform.COVER, Platform.FAN, Platform.LIGHT, Platform.SCENE, Platform.SWITCH]

async def async_setup(hass: HomeAssistant, base_config: ConfigType) -> bool:
    """Set up the Lutron component."""
    if DOMAIN in base_config:
        bridge_configs = base_config[DOMAIN]
        for config in bridge_configs:
            hass.async_create_task(hass.config_entries.flow.async_init(DOMAIN, context={'source': config_entries.SOURCE_IMPORT}, data={CONF_HOST: config[CONF_HOST], CONF_KEYFILE: config[CONF_KEYFILE], CONF_CERTFILE: config[CONF_CERTFILE], CONF_CA_CERTS: config[CONF_CA_CERTS]}))
    return True

async def _async_migrate_unique_ids(hass: HomeAssistant, entry: LutronCasetaConfigEntry) -> None:
    """Migrate entities since the occupancygroup were not actually unique."""
    dev_reg = dr.async_get(hass)
    bridge_unique_id = entry.unique_id

    @callback
    def _async_migrator(entity_entry: er.RegistryEntry) -> Optional[Dict[str, str]]:
        if not (unique_id := entity_entry.unique_id):
            return None
        if not unique_id.startswith('occupancygroup_') or unique_id.startswith(f'occupancygroup_{bridge_unique_id}'):
            return None
        sensor_id = unique_id.split('_')[1]
        new_unique_id = f'occupancygroup_{bridge_unique_id}_{sensor_id}'
        if (dev_entry := dev_reg.async_get_device(identifiers={(DOMAIN, unique_id)})):
            dev_reg.async_update_device(dev_entry.id, new_identifiers={(DOMAIN, new_unique_id)})
        return {'new_unique_id': f'occupancygroup_{bridge_unique_id}_{sensor_id}'}
    await er.async_migrate_entries(hass, entry.entry_id, _async_migrator)

async def async_setup_entry(hass: HomeAssistant, entry: LutronCasetaConfigEntry) -> bool:
    """Set up a bridge from a config entry."""
    entry_id = entry.entry_id
    host = entry.data[CONF_HOST]
    keyfile = hass.config.path(entry.data[CONF_KEYFILE])
    certfile = hass.config.path(entry.data[CONF_CERTFILE])
    ca_certs = hass.config.path(entry.data[CONF_CA_CERTS])
    bridge: Optional[Smartbridge] = None
    try:
        bridge = Smartbridge.create_tls(hostname=host, keyfile=keyfile, certfile=certfile, ca_certs=ca_certs)
    except ssl.SSLError:
        _LOGGER.error('Invalid certificate used to connect to bridge at %s', host)
        return False
    timed_out = True
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(BRIDGE_TIMEOUT):
            await bridge.connect()
            timed_out = False
    if timed_out or not bridge.is_connected():
        await bridge.close()
        if timed_out:
            raise ConfigEntryNotReady(f'Timed out while trying to connect to {host}')
        if not bridge.is_connected():
            raise ConfigEntryNotReady(f'Cannot connect to {host}')
    _LOGGER.debug('Connected to Lutron Caseta bridge via LEAP at %s', host)
    await _async_migrate_unique_ids(hass, entry)
    bridge_devices = bridge.get_devices()
    bridge_device = bridge_devices[BRIDGE_DEVICE_ID]
    if not entry.unique_id:
        hass.config_entries.async_update_entry(entry, unique_id=serial_to_unique_id(bridge_device['serial']))
    _async_register_bridge_device(hass, entry_id, bridge_device, bridge)
    keypad_data = _async_setup_keypads(hass, entry_id, bridge, bridge_device)
    entry.runtime_data = LutronCasetaData(bridge, bridge_device, keypad_data)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

@callback
def _async_register_bridge_device(hass: HomeAssistant, config_entry_id: str, bridge_device: Dict[str, Any], bridge: Smartbridge) -> None:
    """Register the bridge device in the device registry."""
    device_registry = dr.async_get(hass)
    device_args = DeviceInfo(name=bridge_device['name'], manufacturer=MANUFACTURER, identifiers={(DOMAIN, bridge_device['serial'])}, model=f"{bridge_device['model']} ({bridge_device['type']})", via_device=(DOMAIN, bridge_device['serial']), configuration_url='https://device-login.lutron.com')
    area = area_name_from_id(bridge.areas, bridge_device['area'])
    if area != UNASSIGNED_AREA:
        device_args['suggested_area'] = area
    device_registry.async_get_or_create(**device_args, config_entry_id=config_entry_id)

@callback
def _async_setup_keypads(hass: HomeAssistant, config_entry_id: str, bridge: Smartbridge, bridge_device: Dict[str, Any]) -> LutronKeypadData:
    """Register keypad devices (Keypads and Pico Remotes) in the device registry."""
    device_registry = dr.async_get(hass)
    bridge_devices = bridge.get_devices()
    bridge_buttons = bridge.buttons
    dr_device_id_to_keypad: Dict[str, LutronKeypad] = {}
    keypads: Dict[int, LutronKeypad] = {}
    keypad_buttons: Dict[int, LutronButton] = {}
    keypad_button_names_to_leap: Dict[int, Dict[str, int]] = {}
    leap_to_keypad_button_names: Dict[int, Dict[int, str]] = {}
    for bridge_button in bridge_buttons.values():
        parent_device = cast(str, bridge_button['parent_device'])
        bridge_keypad = bridge_devices[parent_device]
        keypad_lutron_device_id = cast(int, bridge_keypad['device_id'])
        button_lutron_device_id = cast(int, bridge_button['device_id'])
        leap_button_number = cast(int, bridge_button['button_number'])
        button_led_device_id = None
        if 'button_led' in bridge_button:
            button_led_device_id = cast(str, bridge_button['button_led'])
        if not (keypad := keypads.get(keypad_lutron_device_id)):
            keypad = keypads[keypad_lutron_device_id] = _async_build_lutron_keypad(bridge, bridge_device, bridge_keypad, keypad_lutron_device_id)
            dr_device = device_registry.async_get_or_create(**keypad['device_info'], config_entry_id=config_entry_id)
            keypad[LUTRON_KEYPAD_DEVICE_REGISTRY_DEVICE_ID] = dr_device.id
            dr_device_id_to_keypad[dr_device.id] = keypad
        button_name = _get_button_name(keypad, bridge_button)
        keypad_lutron_device_id = keypad[LUTRON_KEYPAD_LUTRON_DEVICE_ID]
        keypad_buttons[button_lutron_device_id] = LutronButton(lutron_device_id=button_lutron_device_id, leap_button_number=leap_button_number, button_name=button_name, led_device_id=button_led_device_id, parent_keypad=keypad_lutron_device_id)
        keypad[LUTRON_KEYPAD_BUTTONS].append(button_lutron_device_id)
        button_name_to_leap = keypad_button_names_to_leap.setdefault(keypad_lutron_device_id, {})
        button_name_to_leap[button_name] = leap_button_number
        leap_to_button_name = leap_to_keypad_button_names.setdefault(keypad_lutron_device_id, {})
        leap_to_button_name[leap_button_number] = button_name
    keypad_trigger_schemas = _async_build_trigger_schemas(keypad_button_names_to_leap)
    _async_subscribe_keypad_events(hass=hass, bridge=bridge, keypads=keypads, keypad_buttons=keypad_buttons, leap_to_keypad_button_names=leap_to_keypad_button_names)
    return LutronKeypadData(dr_device_id_to_keypad, keypads, keypad_buttons, keypad_button_names_to_leap, keypad_trigger_schemas)

@callback
def _async_build_trigger_schemas(keypad_button_names_to_leap: Dict[int, Dict[str, int]]) -> Dict[int, vol.Schema]:
    """Build device trigger schemas."""
    return {keypad_id: LUTRON_BUTTON_TRIGGER_SCHEMA.extend({vol.Required(CONF_SUBTYPE): vol.In(keypad_button_names_to_leap[keypad_id])}) for keypad_id in keypad_button_names_to_leap}

@callback
def _async_build_lutron_keypad(bridge: Smartbridge, bridge_device: Dict[str, Any], bridge_keypad: Dict[str, Any], keypad_device_id: int) -> LutronKeypad:
    area_name = area_name_from_id(bridge.areas, bridge_keypad['area'])
    keypad_name = bridge_keypad['name'].split('_')[-1]
    keypad_serial = _handle_none_keypad_serial(bridge_keypad, bridge_device['serial'])
    device_info = DeviceInfo(name=f'{area_name} {keypad_name}', manufacturer=MANUFACTURER, identifiers={(DOMAIN, keypad_serial)}, model=f"{bridge_keypad['model']} ({bridge_keypad['type']})", via_device=(DOMAIN, bridge_device['serial']))
    if area_name != UNASSIGNED_AREA:
        device_info['suggested_area'] = area_name
    return LutronKeypad(lutron_device_id=keypad_device_id, dr_device_id='', area_id=bridge_keypad['area'], area_name=area_name, name=keypad_name, serial=keypad_serial, device_info=device_info, model=bridge_keypad['model'], type=bridge_keypad['type'], buttons=[])

def _get_button_name(keypad: LutronKeypad, bridge_button: Dict[str, Any]) -> str:
    """Get the LEAP button name and check for override."""
    button_number = bridge_button['button_number']
    button_name = bridge_button.get('device_name')
    if button_name is None:
        return _get_button_name_from_triggers(keypad, button_number)
    keypad_model = keypad[LUTRON_KEYPAD_MODEL]
    if (keypad_model_override := KEYPAD_LEAP_BUTTON_NAME_OVERRIDE.get(keypad_model)):
        if (alt_button_name := keypad_model_override.get(button_number)):
            return alt_button_name
    return button_name

def _get_button_name_from_triggers(keypad: LutronKeypad, button_number: int) -> str:
    """Retrieve the caseta button name from device triggers."""
    button_number_map = LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP.get(keypad['type'], {})
    return button_number_map.get(button_number, f'button {button_number}').replace('_', ' ').title()

def _handle_none_keypad_serial(keypad_device: Dict[str, Any], bridge_serial: str) -> str:
    return keypad_device['serial'] or f"{bridge_serial}_{keypad_device['device_id']}"

@callback
def async_get_lip_button(device_type: str, leap_button: int) -> Optional[int]:
    """Get the LIP button for a given LEAP button."""
    if (lip_buttons_name_to_num := DEVICE_TYPE_SUBTYPE_MAP_TO_LIP.get(device_type)) is None or (leap_button_num_to_name := LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP.get(device_type)) is None:
        return None
    return lip_buttons_name_to_num[leap_button_num_to_name[leap_button]]

@callback
def _async_subscribe_keypad_events(hass: HomeAssistant, bridge: Smartbridge, keypads: Dict[int, LutronKeypad], keypad_buttons: Dict[int, LutronButton], leap_to_keypad_button_names: Dict[int, Dict[int, str]]) -> None:
    """Subscribe to lutron events."""

    @callback
    def _async_button_event(button_id: int, event_type: str) -> None:
        if not (button := keypad_buttons.get(button_id)) or not (keypad := keypads.get(button['parent_keypad'])):
            return
        if event_type == BUTTON_STATUS_PRESSED:
            action = ACTION_PRESS
        else:
            action = ACTION_RELEASE
        keypad_type = keypad[LUTRON_KEYPAD_TYPE]
        keypad_device_id = keypad[LUTRON_KEYPAD_LUTRON_DEVICE_ID]
        leap_button_number = button[LUTRON_BUTTON_LEAP_BUTTON_NUMBER]
        lip_button_number = async_get_lip_button(keypad_type, leap_button_number)
        button_type = LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP.get(keypad_type, leap_to_keypad_button_names[keypad_device_id])[leap_button_number]
        hass.bus.async_fire(LUTRON_CASETA_BUTTON_EVENT, {ATTR_SERIAL: keypad[LUTRON_KEYPAD_SERIAL], ATTR_TYPE: keypad_type, ATTR_BUTTON_NUMBER: lip_button_number, ATTR_LEAP_BUTTON_NUMBER: leap_button_number, ATTR_DEVICE_NAME: keypad[LUTRON_KEYPAD_NAME], ATTR_DEVICE_ID: keypad[LUTRON_KEYPAD_DEVICE_REGISTRY_DEVICE_ID], ATTR_AREA_NAME: keypad[LUTRON_KEYPAD_AREA_NAME], ATTR_BUTTON_TYPE: button_type, ATTR_ACTION: action})
    for button_id in keypad_buttons:
        bridge.add_button_subscriber(str(button_id), lambda event_type, button_id=button_id: _async_button_event(button_id, event_type))

async def async_unload_entry(hass: HomeAssistant, entry: LutronCasetaConfigEntry) -> bool:
    """Unload the bridge from a config entry."""
    data = entry.runtime_data
    await data.bridge.close()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

def _id_to_identifier(lutron_id: str) -> Tuple[str, str]:
    """Convert a lutron caseta identifier to a device identifier."""
    return (DOMAIN, lutron_id)

async def async_remove_config_entry_device(hass: HomeAssistant, entry: LutronCasetaConfigEntry, device_entry: DeviceEntry) -> bool:
    """Remove lutron_caseta config entry from a device."""
    data = entry.runtime_data
    bridge = data.bridge
    devices = bridge.get_devices()
    buttons = bridge.buttons
    occupancy_groups = bridge.occupancy_groups
    bridge_device = devices[BRIDGE_DEVICE_ID]
    bridge_unique_id = serial_to_unique_id(bridge_device['serial'])
    all_identifiers = {_id_to_identifier(bridge_unique_id), *(_id_to_identifier(f"occupancygroup_{bridge_unique_id}_{device['occupancy_group_id']}") for device in occupancy_groups.values()), *(_id_to_