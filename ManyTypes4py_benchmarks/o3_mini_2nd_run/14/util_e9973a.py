"""Collection of useful functions for the HomeKit component."""
from __future__ import annotations
import io
import ipaddress
import logging
import os
import re
import secrets
import socket
from typing import Any, cast, Dict, List, Optional, Set, Tuple
from pyhap.accessory import Accessory
import pyqrcode
import voluptuous as vol
from homeassistant.components import binary_sensor, media_player, persistent_notification, sensor
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.event import DOMAIN as EVENT_DOMAIN
from homeassistant.components.lock import DOMAIN as LOCK_DOMAIN
from homeassistant.components.media_player import (
    DOMAIN as MEDIA_PLAYER_DOMAIN,
    MediaPlayerDeviceClass,
    MediaPlayerEntityFeature,
)
from homeassistant.components.remote import DOMAIN as REMOTE_DOMAIN, RemoteEntityFeature
from homeassistant.const import (
    ATTR_CODE,
    ATTR_DEVICE_CLASS,
    ATTR_SUPPORTED_FEATURES,
    CONF_NAME,
    CONF_PORT,
    CONF_TYPE,
    UnitOfTemperature,
)
from homeassistant.core import Event, HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.storage import STORAGE_DIR
from homeassistant.util.unit_conversion import TemperatureConverter
from .const import (
    AUDIO_CODEC_COPY,
    AUDIO_CODEC_OPUS,
    CONF_AUDIO_CODEC,
    CONF_AUDIO_MAP,
    CONF_AUDIO_PACKET_SIZE,
    CONF_FEATURE,
    CONF_FEATURE_LIST,
    CONF_LINKED_BATTERY_CHARGING_SENSOR,
    CONF_LINKED_BATTERY_SENSOR,
    CONF_LINKED_DOORBELL_SENSOR,
    CONF_LINKED_HUMIDITY_SENSOR,
    CONF_LINKED_MOTION_SENSOR,
    CONF_LINKED_OBSTRUCTION_SENSOR,
    CONF_LOW_BATTERY_THRESHOLD,
    CONF_MAX_FPS,
    CONF_MAX_HEIGHT,
    CONF_MAX_WIDTH,
    CONF_STREAM_ADDRESS,
    CONF_STREAM_COUNT,
    CONF_STREAM_SOURCE,
    CONF_SUPPORT_AUDIO,
    CONF_THRESHOLD_CO,
    CONF_THRESHOLD_CO2,
    CONF_VIDEO_CODEC,
    CONF_VIDEO_MAP,
    CONF_VIDEO_PACKET_SIZE,
    CONF_VIDEO_PROFILE_NAMES,
    DEFAULT_AUDIO_CODEC,
    DEFAULT_AUDIO_MAP,
    DEFAULT_AUDIO_PACKET_SIZE,
    DEFAULT_LOW_BATTERY_THRESHOLD,
    DEFAULT_MAX_FPS,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_WIDTH,
    DEFAULT_STREAM_COUNT,
    DEFAULT_SUPPORT_AUDIO,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_MAP,
    DEFAULT_VIDEO_PACKET_SIZE,
    DEFAULT_VIDEO_PROFILE_NAMES,
    DOMAIN,
    FEATURE_ON_OFF,
    FEATURE_PLAY_PAUSE,
    FEATURE_PLAY_STOP,
    FEATURE_TOGGLE_MUTE,
    MAX_NAME_LENGTH,
    TYPE_FAUCET,
    TYPE_OUTLET,
    TYPE_SHOWER,
    TYPE_SPRINKLER,
    TYPE_SWITCH,
    TYPE_VALVE,
)
from .models import HomeKitConfigEntry

_LOGGER = logging.getLogger(__name__)
NUMBERS_ONLY_RE = re.compile('[^\\d.]+')
VERSION_RE = re.compile('([0-9]+)(\\.[0-9]+)?(\\.[0-9]+)?')
INVALID_END_CHARS = '-_ '
MAX_VERSION_PART = 2 ** 32 - 1
MAX_PORT = 65535
VALID_VIDEO_CODECS: List[str] = [VIDEO_CODEC_LIBX264, VIDEO_CODEC_H264_OMX, VIDEO_CODEC_H264_V4L2M2M, AUDIO_CODEC_COPY]
VALID_AUDIO_CODECS: List[str] = [AUDIO_CODEC_OPUS, VIDEO_CODEC_COPY]
BASIC_INFO_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): cv.string,
        vol.Optional(CONF_LINKED_BATTERY_SENSOR): cv.entity_domain(sensor.DOMAIN),
        vol.Optional(CONF_LINKED_BATTERY_CHARGING_SENSOR): cv.entity_domain(binary_sensor.DOMAIN),
        vol.Optional(CONF_LOW_BATTERY_THRESHOLD, default=DEFAULT_LOW_BATTERY_THRESHOLD): cv.positive_int,
    }
)
FEATURE_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_FEATURE_LIST, default=None): cv.ensure_list})
CAMERA_SCHEMA = BASIC_INFO_SCHEMA.extend(
    {
        vol.Optional(CONF_STREAM_ADDRESS): vol.All(ipaddress.ip_address, cv.string),
        vol.Optional(CONF_STREAM_SOURCE): cv.string,
        vol.Optional(CONF_AUDIO_CODEC, default=DEFAULT_AUDIO_CODEC): vol.In(VALID_AUDIO_CODECS),
        vol.Optional(CONF_SUPPORT_AUDIO, default=DEFAULT_SUPPORT_AUDIO): cv.boolean,
        vol.Optional(CONF_MAX_WIDTH, default=DEFAULT_MAX_WIDTH): cv.positive_int,
        vol.Optional(CONF_MAX_HEIGHT, default=DEFAULT_MAX_HEIGHT): cv.positive_int,
        vol.Optional(CONF_MAX_FPS, default=DEFAULT_MAX_FPS): cv.positive_int,
        vol.Optional(CONF_AUDIO_MAP, default=DEFAULT_AUDIO_MAP): cv.string,
        vol.Optional(CONF_VIDEO_MAP, default=DEFAULT_VIDEO_MAP): cv.string,
        vol.Optional(CONF_STREAM_COUNT, default=DEFAULT_STREAM_COUNT): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
        vol.Optional(CONF_VIDEO_CODEC, default=DEFAULT_VIDEO_CODEC): vol.In(VALID_VIDEO_CODECS),
        vol.Optional(CONF_VIDEO_PROFILE_NAMES, default=DEFAULT_VIDEO_PROFILE_NAMES): [cv.string],
        vol.Optional(CONF_AUDIO_PACKET_SIZE, default=DEFAULT_AUDIO_PACKET_SIZE): cv.positive_int,
        vol.Optional(CONF_VIDEO_PACKET_SIZE, default=DEFAULT_VIDEO_PACKET_SIZE): cv.positive_int,
        vol.Optional(CONF_LINKED_MOTION_SENSOR): cv.entity_domain([binary_sensor.DOMAIN, EVENT_DOMAIN]),
        vol.Optional(CONF_LINKED_DOORBELL_SENSOR): cv.entity_domain([binary_sensor.DOMAIN, EVENT_DOMAIN]),
    }
)
HUMIDIFIER_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_LINKED_HUMIDITY_SENSOR): cv.entity_domain(sensor.DOMAIN)})
COVER_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_LINKED_OBSTRUCTION_SENSOR): cv.entity_domain(binary_sensor.DOMAIN)})
CODE_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(ATTR_CODE, default=None): vol.Any(None, cv.string)})
LOCK_SCHEMA = CODE_SCHEMA.extend({vol.Optional(CONF_LINKED_DOORBELL_SENSOR): cv.entity_domain([binary_sensor.DOMAIN, EVENT_DOMAIN])})
MEDIA_PLAYER_SCHEMA = vol.Schema({vol.Required(CONF_FEATURE): vol.All(cv.string, vol.In((FEATURE_ON_OFF, FEATURE_PLAY_PAUSE, FEATURE_PLAY_STOP, FEATURE_TOGGLE_MUTE)))})
SWITCH_TYPE_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_TYPE, default=TYPE_SWITCH): vol.All(cv.string, vol.In((TYPE_FAUCET, TYPE_OUTLET, TYPE_SHOWER, TYPE_SPRINKLER, TYPE_SWITCH, TYPE_VALVE)))})
SENSOR_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_THRESHOLD_CO): vol.Any(None, cv.positive_int), vol.Optional(CONF_THRESHOLD_CO2): vol.Any(None, cv.positive_int)})
HOMEKIT_CHAR_TRANSLATIONS: Dict[int, str] = {
    0: ' ',
    10: ' ',
    13: ' ',
    33: '-',
    34: ' ',
    36: '-',
    37: '-',
    40: '-',
    41: '-',
    42: '-',
    43: '-',
    47: '-',
    58: '-',
    59: '-',
    60: '-',
    61: '-',
    62: '-',
    63: '-',
    64: '-',
    91: '-',
    92: '-',
    93: '-',
    94: '-',
    95: ' ',
    96: '-',
    123: '-',
    124: '-',
    125: '-',
    126: '-',
    127: '-',
}


def validate_entity_config(values: Any) -> Dict[str, Dict[str, Any]]:
    """Validate config entry for CONF_ENTITY."""
    if not isinstance(values, dict):
        raise vol.Invalid('expected a dictionary')
    entities: Dict[str, Dict[str, Any]] = {}
    for entity_id, config in values.items():
        entity: str = cv.entity_id(entity_id)
        domain, _ = split_entity_id(entity)
        if not isinstance(config, dict):
            raise vol.Invalid(f'The configuration for {entity} must be a dictionary.')
        if domain == 'alarm_control_panel':
            config = CODE_SCHEMA(config)
        elif domain == media_player.const.DOMAIN:
            config = FEATURE_SCHEMA(config)
            feature_list: Dict[str, Any] = {}
            for feature in config[CONF_FEATURE_LIST]:
                params: Dict[str, Any] = MEDIA_PLAYER_SCHEMA(feature)
                key: str = params.pop(CONF_FEATURE)
                if key in feature_list:
                    raise vol.Invalid(f'A feature can be added only once for {entity}')
                feature_list[key] = params
            config[CONF_FEATURE_LIST] = feature_list
        elif domain == 'camera':
            config = CAMERA_SCHEMA(config)
        elif domain == 'lock':
            config = LOCK_SCHEMA(config)
        elif domain == 'switch':
            config = SWITCH_TYPE_SCHEMA(config)
        elif domain == 'humidifier':
            config = HUMIDIFIER_SCHEMA(config)
        elif domain == 'cover':
            config = COVER_SCHEMA(config)
        elif domain == 'sensor':
            config = SENSOR_SCHEMA(config)
        else:
            config = BASIC_INFO_SCHEMA(config)
        entities[entity] = config
    return entities


def get_media_player_features(state: State) -> List[str]:
    """Determine features for media players."""
    features: int = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0)
    supported_modes: List[str] = []
    if features & (MediaPlayerEntityFeature.TURN_ON | MediaPlayerEntityFeature.TURN_OFF):
        supported_modes.append(FEATURE_ON_OFF)
    if features & (MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.PAUSE):
        supported_modes.append(FEATURE_PLAY_PAUSE)
    if features & (MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.STOP):
        supported_modes.append(FEATURE_PLAY_STOP)
    if features & MediaPlayerEntityFeature.VOLUME_MUTE:
        supported_modes.append(FEATURE_TOGGLE_MUTE)
    return supported_modes


def validate_media_player_features(state: State, feature_list: Optional[List[str]]) -> bool:
    """Validate features for media players."""
    if not (supported_modes := get_media_player_features(state)):
        _LOGGER.error('%s does not support any media_player features', state.entity_id)
        return False
    if not feature_list:
        return True
    error_list: List[str] = [feature for feature in feature_list if feature not in supported_modes]
    if error_list:
        _LOGGER.error('%s does not support media_player features: %s', state.entity_id, error_list)
        return False
    return True


def async_show_setup_message(
    hass: HomeAssistant, entry_id: str, bridge_name: str, pincode: bytes, uri: str
) -> None:
    """Display persistent notification with setup information."""
    pin: str = pincode.decode()
    _LOGGER.info('Pincode: %s', pin)
    buffer: io.BytesIO = io.BytesIO()
    url = pyqrcode.create(uri)
    url.svg(buffer, scale=5, module_color='#000', background='#FFF')
    pairing_secret: str = secrets.token_hex(32)
    entry: HomeKitConfigEntry = cast(HomeKitConfigEntry, hass.config_entries.async_get_entry(entry_id))
    entry_data = entry.runtime_data
    entry_data.pairing_qr = buffer.getvalue()
    entry_data.pairing_qr_secret = pairing_secret
    message: str = (
        f'To set up {bridge_name} in the Home App, scan the QR code or enter the following code:\n'
        f'### {pin}\n'
        f'![image](/api/homekit/pairingqr?{entry_id}-{pairing_secret})'
    )
    persistent_notification.async_create(hass, message, 'HomeKit Pairing', entry_id)


def async_dismiss_setup_message(hass: HomeAssistant, entry_id: str) -> None:
    """Dismiss persistent notification and remove QR code."""
    persistent_notification.async_dismiss(hass, entry_id)


def convert_to_float(state: Any) -> Optional[float]:
    """Return float of state, catch errors."""
    try:
        return float(state)
    except (ValueError, TypeError):
        return None


def coerce_int(state: Any) -> int:
    """Return int."""
    try:
        return int(state)
    except (ValueError, TypeError):
        return 0


def cleanup_name_for_homekit(name: Optional[str]) -> str:
    """Ensure the name of the device will not crash homekit."""
    if name is None:
        return 'None'
    return name.translate(HOMEKIT_CHAR_TRANSLATIONS).lstrip(INVALID_END_CHARS)[:MAX_NAME_LENGTH].rstrip(INVALID_END_CHARS)


def temperature_to_homekit(temperature: float, unit: str) -> float:
    """Convert temperature to Celsius for HomeKit."""
    return TemperatureConverter.convert(temperature, unit, UnitOfTemperature.CELSIUS)


def temperature_to_states(temperature: float, unit: str) -> float:
    """Convert temperature back from Celsius to Home Assistant unit."""
    return TemperatureConverter.convert(temperature, UnitOfTemperature.CELSIUS, unit)


def density_to_air_quality(density: float) -> int:
    """Map PM2.5 µg/m3 density to HomeKit AirQuality level."""
    if density <= 9:
        return 1
    if density <= 35.4:
        return 2
    if density <= 55.4:
        return 3
    if density <= 125.4:
        return 4
    return 5


def density_to_air_quality_pm10(density: float) -> int:
    """Map PM10 µg/m3 density to HomeKit AirQuality level."""
    if density <= 54:
        return 1
    if density <= 154:
        return 2
    if density <= 254:
        return 3
    if density <= 354:
        return 4
    return 5


def density_to_air_quality_nitrogen_dioxide(density: float) -> int:
    """Map nitrogen dioxide µg/m3 to HomeKit AirQuality level."""
    if density <= 30:
        return 1
    if density <= 60:
        return 2
    if density <= 80:
        return 3
    if density <= 90:
        return 4
    return 5


def density_to_air_quality_voc(density: float) -> int:
    """Map VOCs µg/m3 to HomeKit AirQuality level.

    The VOC mappings use the IAQ guidelines for Europe released by the WHO (World Health Organization).
    Referenced from Sensirion_Gas_Sensors_SGP3x_TVOC_Concept.pdf
    https://github.com/paulvha/svm30/blob/master/extras/Sensirion_Gas_Sensors_SGP3x_TVOC_Concept.pdf
    """
    if density <= 250:
        return 1
    if density <= 500:
        return 2
    if density <= 1000:
        return 3
    if density <= 3000:
        return 4
    return 5


def get_persist_filename_for_entry_id(entry_id: str) -> str:
    """Determine the filename of the homekit state file."""
    return f'{DOMAIN}.{entry_id}.state'


def get_aid_storage_filename_for_entry_id(entry_id: str) -> str:
    """Determine the filename of homekit aid storage file."""
    return f'{DOMAIN}.{entry_id}.aids'


def get_iid_storage_filename_for_entry_id(entry_id: str) -> str:
    """Determine the filename of homekit iid storage file."""
    return f'{DOMAIN}.{entry_id}.iids'


def get_persist_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
    """Determine the path to the homekit state file."""
    return hass.config.path(STORAGE_DIR, get_persist_filename_for_entry_id(entry_id))


def get_aid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
    """Determine the path to the homekit aid storage file."""
    return hass.config.path(STORAGE_DIR, get_aid_storage_filename_for_entry_id(entry_id))


def get_iid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
    """Determine the path to the homekit iid storage file."""
    return hass.config.path(STORAGE_DIR, get_iid_storage_filename_for_entry_id(entry_id))


def _format_version_part(version_part: Any) -> str:
    return str(max(0, min(MAX_VERSION_PART, coerce_int(version_part))))


def format_version(version: Any) -> Optional[str]:
    """Extract the version string in a format homekit can consume."""
    split_ver: str = str(version).replace('-', '.').replace(' ', '.')
    num_only: str = NUMBERS_ONLY_RE.sub('', split_ver)
    if (match := VERSION_RE.search(num_only)) is None:
        return None
    value: str = '.'.join(map(_format_version_part, match.group(0).split('.')))
    return None if _is_zero_but_true(value) else value


def _is_zero_but_true(value: str) -> bool:
    """Zero but true values can crash apple watches."""
    return convert_to_float(value) == 0


def remove_state_files_for_entry_id(hass: HomeAssistant, entry_id: str) -> None:
    """Remove the state files from disk."""
    for path in (
        get_persist_fullpath_for_entry_id(hass, entry_id),
        get_aid_storage_fullpath_for_entry_id(hass, entry_id),
        get_iid_storage_fullpath_for_entry_id(hass, entry_id),
    ):
        if os.path.exists(path):
            os.unlink(path)


def _get_test_socket() -> socket.socket:
    """Create a socket to test binding ports."""
    test_socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_socket.setblocking(False)
    test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return test_socket


@callback
def async_port_is_available(port: int) -> bool:
    """Check to see if a port is available."""
    try:
        _get_test_socket().bind(('', port))
    except OSError:
        return False
    return True


@callback
def async_find_next_available_port(hass: HomeAssistant, start_port: int) -> int:
    """Find the next available port not assigned to a config entry."""
    exclude_ports: Set[int] = {
        entry.data[CONF_PORT]
        for entry in hass.config_entries.async_entries(DOMAIN)
        if CONF_PORT in entry.data
    }
    return _async_find_next_available_port(start_port, exclude_ports)


@callback
def _async_find_next_available_port(start_port: int, exclude_ports: Set[int]) -> int:
    """Find the next available port starting with the given port."""
    test_socket: socket.socket = _get_test_socket()
    for port in range(start_port, MAX_PORT + 1):
        if port in exclude_ports:
            continue
        try:
            test_socket.bind(('', port))
        except OSError:
            if port == MAX_PORT:
                raise
            continue
        else:
            return port
    raise RuntimeError('unreachable')


def pid_is_alive(pid: int) -> bool:
    """Check to see if a process is alive."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def accessory_friendly_name(hass_name: str, accessory: Accessory) -> str:
    """Return the combined name for the accessory.

    The mDNS name and the Home Assistant config entry
    name are usually different which means they need to
    see both to identify the accessory.
    """
    accessory_mdns_name: str = cast(str, accessory.display_name)
    if hass_name.casefold().startswith(accessory_mdns_name.casefold()):
        return hass_name
    if accessory_mdns_name.casefold().startswith(hass_name.casefold()):
        return accessory_mdns_name
    return f'{hass_name} ({accessory_mdns_name})'


def state_needs_accessory_mode(state: State) -> bool:
    """Return if the entity represented by the state must be paired in accessory mode."""
    if state.domain in (CAMERA_DOMAIN, LOCK_DOMAIN):
        return True
    return state.domain == MEDIA_PLAYER_DOMAIN and state.attributes.get(ATTR_DEVICE_CLASS) in (MediaPlayerDeviceClass.TV, MediaPlayerDeviceClass.RECEIVER) or (
        state.domain == REMOTE_DOMAIN and state.attributes.get(ATTR_SUPPORTED_FEATURES, 0) & RemoteEntityFeature.ACTIVITY
    )


def state_changed_event_is_same_state(event: Event) -> bool:
    """Check if a state changed event is the same state."""
    event_data: Dict[str, Any] = event.data
    old_state: Optional[State] = event_data.get('old_state')
    new_state: Optional[State] = event_data.get('new_state')
    return bool(new_state and old_state and (new_state.state == old_state.state))


def get_min_max(value1: float, value2: float) -> Tuple[float, float]:
    """Return the minimum and maximum of two values.

    HomeKit will go unavailable if the min and max are reversed
    so we make sure the min is always the min and the max is always the max
    as any mistakes made in integrations will cause the entire
    bridge to go unavailable.
    """
    return (min(value1, value2), max(value1, value2))
