from __future__ import annotations
import asyncio
import contextlib
from itertools import chain
import logging
import ssl
from typing import Any, cast
from pylutron_caseta import BUTTON_STATUS_PRESSED
from pylutron_caseta.smartbridge import Smartbridge
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import ATTR_DEVICE_ID, CONF_HOST, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv, device_registry as dr, entity_registry as er
from homeassistant.helpers.typing import ConfigType
from .const import ACTION_PRESS, ACTION_RELEASE, ATTR_ACTION, ATTR_AREA_NAME, ATTR_BUTTON_NUMBER, ATTR_BUTTON_TYPE, ATTR_DEVICE_NAME, ATTR_LEAP_BUTTON_NUMBER, ATTR_SERIAL, ATTR_TYPE, BRIDGE_DEVICE_ID, BRIDGE_TIMEOUT, CONF_CA_CERTS, CONF_CERTFILE, CONF_KEYFILE, CONF_SUBTYPE, DOMAIN, LUTRON_CASETA_BUTTON_EVENT, MANUFACTURER, UNASSIGNED_AREA
from .device_trigger import DEVICE_TYPE_SUBTYPE_MAP_TO_LIP, KEYPAD_LEAP_BUTTON_NAME_OVERRIDE, LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP, LUTRON_BUTTON_TRIGGER_SCHEMA
from .models import LUTRON_BUTTON_LEAP_BUTTON_NUMBER, LUTRON_KEYPAD_AREA_NAME, LUTRON_KEYPAD_BUTTONS, LUTRON_KEYPAD_DEVICE_REGISTRY_DEVICE_ID, LUTRON_KEYPAD_LUTRON_DEVICE_ID, LUTRON_KEYPAD_MODEL, LUTRON_KEYPAD_NAME, LUTRON_KEYPAD_SERIAL, LUTRON_KEYPAD_TYPE, LutronButton, LutronCasetaConfigEntry, LutronCasetaData, LutronKeypad, LutronKeypadData
from .util import area_name_from_id, serial_to_unique_id

_LOGGER: logging.Logger = logging.getLogger(__name__)
DATA_BRIDGE_CONFIG: str = 'lutron_caseta_bridges'
CONFIG_SCHEMA: vol.Schema = vol.Schema({DOMAIN: vol.All(cv.ensure_list, [{vol.Required(CONF_HOST): cv.string, vol.Required(CONF_KEYFILE): cv.string, vol.Required(CONF_CERTFILE): cv.string, vol.Required(CONF_CA_CERTS): cv.string}])}, extra=vol.ALLOW_EXTRA)
PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.BUTTON, Platform.COVER, Platform.FAN, Platform.LIGHT, Platform.SCENE, Platform.SWITCH]

async def async_setup(hass: HomeAssistant, base_config: ConfigType) -> bool:
    """Set up the Lutron component."""
    if DOMAIN in base_config:
        bridge_configs: list[dict] = base_config[DOMAIN]
        for config in bridge_configs:
            hass.async_create_task(hass.config_entries.flow.async_init(DOMAIN, context={'source': config_entries.SOURCE_IMPORT}, data={CONF_HOST: config[CONF_HOST], CONF_KEYFILE: config[CONF_KEYFILE], CONF_CERTFILE: config[CONF_CERTFILE], CONF_CA_CERTS: config[CONF_CA_CERTS]}))
    return True

async def _async_migrate_unique_ids(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> None:
    """Migrate entities since the occupancygroup were not actually unique."""
    dev_reg: dr.DeviceRegistry = dr.async_get(hass)
    bridge_unique_id: str = entry.unique_id

    @callback
    def _async_migrator(entity_entry: er.EntityEntry) -> dict | None:
        if not (unique_id := entity_entry.unique_id):
            return None
        if not unique_id.startswith('occupancygroup_') or unique_id.startswith(f'occupancygroup_{bridge_unique_id}'):
            return None
        sensor_id: str = unique_id.split('_')[1]
        new_unique_id: str = f'occupancygroup_{bridge_unique_id}_{sensor_id}'
        if (dev_entry := dev_reg.async_get_device(identifiers={(DOMAIN, unique_id)})):
            dev_reg.async_update_device(dev_entry.id, new_identifiers={(DOMAIN, new_unique_id)})
        return {'new_unique_id': f'occupancygroup_{bridge_unique_id}_{sensor_id}'}
    await er.async_migrate_entries(hass, entry.entry_id, _async_migrator)

async def async_setup_entry(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    """Set up a bridge from a config entry."""
    entry_id: str = entry.entry_id
    host: str = entry.data[CONF_HOST]
    keyfile: str = hass.config.path(entry.data[CONF_KEYFILE])
    certfile: str = hass.config.path(entry.data[CONF_CERTFILE])
    ca_certs: str = hass.config.path(entry.data[CONF_CA_CERTS])
    bridge: Smartbridge | None = None
    try:
        bridge = Smartbridge.create_tls(hostname=host, keyfile=keyfile, certfile=certfile, ca_certs=ca_certs)
    except ssl.SSLError:
        _LOGGER.error('Invalid certificate used to connect to bridge at %s', host)
        return False
    timed_out: bool = True
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
    bridge_devices: dict[str, dict] = bridge.get_devices()
    bridge_device: dict = bridge_devices[BRIDGE_DEVICE_ID]
    if not entry.unique_id:
        hass.config_entries.async_update_entry(entry, unique_id=serial_to_unique_id(bridge_device['serial']))
    _async_register_bridge_device(hass, entry_id, bridge_device, bridge)
    keypad_data: LutronKeypadData = _async_setup_keypads(hass, entry_id, bridge, bridge_device)
    entry.runtime_data: LutronCasetaData = LutronCasetaData(bridge, bridge_device, keypad_data)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

@callback
def _async_register_bridge_device(hass: HomeAssistant, config_entry_id: str, bridge_device: dict, bridge: Smartbridge) -> None:
    """Register the bridge device in the device registry."""
    device_registry: dr.DeviceRegistry = dr.async_get(hass)
    device_args: dict[str, str | int | dict] = DeviceInfo(name=bridge_device['name'], manufacturer=MANUFACTURER, identifiers={(DOMAIN, bridge_device['serial'])}, model=f'{bridge_device["model"]} ({bridge_device["type"]})', via_device=(DOMAIN, bridge_device['serial']), configuration_url='https://device-login.lutron.com')
    area: str = area_name_from_id(bridge.areas, bridge_device['area'])
    if area != UNASSIGNED_AREA:
        device_args['suggested_area'] = area
    device_registry.async_get_or_create(**device_args, config_entry_id=config_entry_id)

@callback
def _async_setup_keypads(hass: HomeAssistant, config_entry_id: str, bridge: Smartbridge, bridge_device: dict) -> LutronKeypadData:
    """Register keypad devices (Keypads and Pico Remotes) in the device registry."""
    device_registry: dr.DeviceRegistry = dr.async_get(hass)
    bridge_devices: dict[str, dict] = bridge.get_devices()
    bridge_buttons: dict[str, dict] = bridge.buttons
    dr_device_id_to_keypad: dict[str, LutronKeypad] = {}
    keypads: dict[int, LutronKeypad] = {}
    keypad_buttons: dict[int, LutronButton] = {}
    keypad_button_names_to_leap: dict[int, dict[str, int]] = {}
    leap_to_keypad_button_names: dict[int, dict[int, str]] = {}
    for bridge_button in bridge_buttons.values():
        parent_device: str = cast(str, bridge_button['parent_device'])
        bridge_keypad: dict = bridge_devices[parent_device]
        keypad_lutron_device_id: int = cast(int, bridge_keypad['device_id'])
        button_lutron_device_id: int = cast(int, bridge_button['device_id'])
        leap_button_number: int = cast(int, bridge_button['button_number'])
        button_led_device_id: str | None = None
        if 'button_led' in bridge_button:
            button_led_device_id = cast(str, bridge_button['button_led'])
        if not (keypad := keypads.get(keypad_lutron_device_id)):
            keypad = keypads[keypad_lutron_device_id] = _async_build_lutron_keypad(bridge, bridge_device, bridge_keypad, keypad_lutron_device_id)
            device_info: DeviceInfo = keypad['device_info']
            if area_name := area_name_from_id(bridge.areas, bridge_keypad['area']):
                device_info['suggested_area'] = area_name
            dr_device = device_registry.async_get_or_create(**device_info, config_entry_id=config_entry_id)
            keypad[LUTRON_KEYPAD_DEVICE_REGISTRY_DEVICE_ID] = dr_device.id
            dr_device_id_to_keypad[dr_device.id] = keypad
        button_name: str = _get_button_name(keypad, bridge_button)
        keypad_lutron_device_id = keypad[LUTRON_KEYPAD_LUTRON_DEVICE_ID]
        keypad_buttons[button_lutron_device_id] = LutronButton(lutron_device_id=button_lutron_device_id, leap_button_number=leap_button_number, button_name=button_name, led_device_id=button_led_device_id, parent_keypad=keypad_lutron_device_id)
        keypad[LUTRON_KEYPAD_BUTTONS].append(button_lutron_device_id)
        button_name_to_leap = keypad_button_names_to_leap.setdefault(keypad_lutron_device_id, {})
        button_name_to_leap[button_name] = leap_button_number
        leap_to_button_name = leap_to_keypad_button_names.setdefault(keypad_lutron_device_id, {})
        leap_to_button_name[leap_button_number] = button_name
    keypad_trigger_schemas: dict[int, vol.Schema] = _async_build_trigger_schemas(keypad_button_names_to_leap)
    _async_subscribe_keypad_events(hass, bridge, keypads, keypad_buttons, leap_to_keypad_button_names)
    return LutronKeypadData(dr_device_id_to_keypad, keypads, keypad_buttons, keypad_button_names_to_leap, keypad_trigger_schemas)

@callback
def _async_build_trigger_schemas(keypad_button_names_to_leap: dict[int, dict[str, int]]) -> dict[int, vol.Schema]:
    """Build device trigger schemas."""
    return {keypad_id: LUTRON_BUTTON_TRIGGER_SCHEMA.extend({vol.Required(CONF_SUBTYPE): vol.In(keypad_button_names_to_leap[keypad_id])}) for keypad_id in keypad_button_names_to_leap}

@callback
def _async_build_lutron_keypad(bridge: Smartbridge, bridge_device: dict, bridge_keypad: dict, keypad_device_id: int) -> LutronKeypad:
    """Build a Lutron Keypad."""
    area_name: str = area_name_from_id(bridge.areas, bridge_keypad['area'])
    keypad_name: str = bridge_keypad['name'].split('_')[-1]
    keypad_serial: str = _handle_none_keypad_serial(bridge_keypad, bridge_device['serial'])
    device_info: DeviceInfo = DeviceInfo(name=f'{area_name} {keypad_name}', manufacturer=MANUFACTURER, identifiers={(DOMAIN, keypad_serial)}, model=f'{bridge_keypad["model"]} ({bridge_keypad["type"]})', via_device=(DOMAIN, bridge_device['serial']))
    if area_name != UNASSIGNED_AREA:
        device_info['suggested_area'] = area_name
    return LutronKeypad(lutron_device_id=keypad_device_id, dr_device_id='', area_id=bridge_keypad['area'], area_name=area_name, name=keypad_name, serial=keypad_serial, device_info=device_info, model=bridge_keypad['model'], type=bridge_keypad['type'], buttons=[])

def _get_button_name(keypad: LutronKeypad, bridge_button: dict) -> str:
    """Get the LEAP button name and check for override."""
    button_number: int = bridge_button['button_number']
    button_name: str | None = bridge_button.get('device_name')
    if button_name is None:
        return _get_button_name_from_triggers(keypad, button_number)
    keypad_model: str = keypad[LUTRON_KEYPAD_MODEL]
    if (keypad_model_override := KEYPAD_LEAP_BUTTON_NAME_OVERRIDE.get(keypad_model)):
        if (alt_button_name := keypad_model_override.get(button_number)):
            return alt_button_name
    return button_name

def _get_button_name_from_triggers(keypad: LutronKeypad, button_number: int) -> str:
    """Retrieve the caseta button name from device triggers."""
    button_number_map: dict[int, str] = LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP.get(keypad[LUTRON_KEYPAD_TYPE], {})
    return button_number_map.get(button_number, f'button {button_number}').replace('_', ' ').title()

def _handle_none_keypad_serial(keypad_device: dict, bridge_serial: str) -> str:
    """Handle a keypad device with no serial."""
    return keypad_device['serial'] or f'{bridge_serial}_{keypad_device["device_id"]}'

@callback
def async_get_lip_button(device_type: str, leap_button: int) -> str | None:
    """Get the LIP button for a given LEAP button."""
    if (lip_buttons_name_to_num := DEVICE_TYPE_SUBTYPE_MAP_TO_LIP.get(device_type)) is None or (leap_button_num_to_name := LEAP_TO_DEVICE_TYPE_SUBTYPE_MAP.get(device_type)) is None:
        return None
    return lip_buttons_name_to_num[leap_button_num_to_name[leap_button]]

async def async_unload_entry(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    """Unload the bridge from a config entry."""
    data: LutronCasetaData = entry.runtime_data
    await data.bridge.close()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

async def async_remove_config_entry_device(hass: HomeAssistant, entry: config_entries.ConfigEntry, device_entry: er.EntityEntry) -> bool:
    """Remove lutron_caseta config entry from a device."""
    data: LutronCasetaData = entry.runtime_data
    bridge: Smartbridge | None = data.bridge
    devices: dict[str, dict] = bridge.get_devices()
    buttons: dict[str, dict] = bridge.buttons
    occupancy_groups: dict[str, dict] = bridge.occupancy_groups
    bridge_device: dict = devices[BRIDGE_DEVICE_ID]
    bridge_unique_id: str = serial_to_unique_id(bridge_device['serial'])
    all_identifiers: set[tuple[str, str]] = {_id_to_identifier(bridge_unique_id), *(_id_to_identifier(f'occupancygroup_{bridge_unique_id}_{device["occupancy_group_id"]}') for device in occupancy_groups.values()), *(_id_to_identifier(device['serial']) for device in chain(devices.values(), buttons.values()))}
    return not any((identifier for identifier in device_entry.identifiers if identifier in all_identifiers)
