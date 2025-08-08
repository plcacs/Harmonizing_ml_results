from __future__ import annotations
import io
import ipaddress
import logging
import os
import re
import secrets
import socket
from typing import Any, cast
from pyhap.accessory import Accessory
import pyqrcode
import voluptuous as vol
from homeassistant.components import binary_sensor, media_player, persistent_notification, sensor
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.event import DOMAIN as EVENT_DOMAIN
from homeassistant.components.lock import DOMAIN as LOCK_DOMAIN
from homeassistant.components.media_player import DOMAIN as MEDIA_PLAYER_DOMAIN, MediaPlayerDeviceClass, MediaPlayerEntityFeature
from homeassistant.components.remote import DOMAIN as REMOTE_DOMAIN, RemoteEntityFeature
from homeassistant.const import ATTR_CODE, ATTR_DEVICE_CLASS, ATTR_SUPPORTED_FEATURES, CONF_NAME, CONF_PORT, CONF_TYPE, UnitOfTemperature
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.storage import STORAGE_DIR
from homeassistant.util.unit_conversion import TemperatureConverter
from .const import AUDIO_CODEC_COPY, AUDIO_CODEC_OPUS, CONF_AUDIO_CODEC, CONF_AUDIO_MAP, CONF_AUDIO_PACKET_SIZE, CONF_FEATURE, CONF_FEATURE_LIST, CONF_LINKED_BATTERY_CHARGING_SENSOR, CONF_LINKED_BATTERY_SENSOR, CONF_LINKED_DOORBELL_SENSOR, CONF_LINKED_HUMIDITY_SENSOR, CONF_LINKED_MOTION_SENSOR, CONF_LINKED_OBSTRUCTION_SENSOR, CONF_LOW_BATTERY_THRESHOLD, CONF_MAX_FPS, CONF_MAX_HEIGHT, CONF_MAX_WIDTH, CONF_STREAM_ADDRESS, CONF_STREAM_COUNT, CONF_STREAM_SOURCE, CONF_SUPPORT_AUDIO, CONF_THRESHOLD_CO, CONF_THRESHOLD_CO2, CONF_VIDEO_CODEC, CONF_VIDEO_MAP, CONF_VIDEO_PACKET_SIZE, CONF_VIDEO_PROFILE_NAMES, DEFAULT_AUDIO_CODEC, DEFAULT_AUDIO_MAP, DEFAULT_AUDIO_PACKET_SIZE, DEFAULT_LOW_BATTERY_THRESHOLD, DEFAULT_MAX_FPS, DEFAULT_MAX_HEIGHT, DEFAULT_MAX_WIDTH, DEFAULT_STREAM_COUNT, DEFAULT_SUPPORT_AUDIO, DEFAULT_VIDEO_CODEC, DEFAULT_VIDEO_MAP, DEFAULT_VIDEO_PACKET_SIZE, DEFAULT_VIDEO_PROFILE_NAMES, DOMAIN, FEATURE_ON_OFF, FEATURE_PLAY_PAUSE, FEATURE_PLAY_STOP, FEATURE_TOGGLE_MUTE, MAX_NAME_LENGTH, TYPE_FAUCET, TYPE_OUTLET, TYPE_SHOWER, TYPE_SPRINKLER, TYPE_SWITCH, TYPE_VALVE, VIDEO_CODEC_COPY, VIDEO_CODEC_H264_OMX, VIDEO_CODEC_H264_V4L2M2M, VIDEO_CODEC_LIBX264
from .models import HomeKitConfigEntry
_LOGGER = logging.getLogger(__name__)
NUMBERS_ONLY_RE = re.compile('[^\\d.]+')
VERSION_RE = re.compile('([0-9]+)(\\.[0-9]+)?(\\.[0-9]+)?')
INVALID_END_CHARS = '-_ '
MAX_VERSION_PART = 2 ** 32 - 1
MAX_PORT = 65535
VALID_VIDEO_CODECS = [VIDEO_CODEC_LIBX264, VIDEO_CODEC_H264_OMX, VIDEO_CODEC_H264_V4L2M2M, AUDIO_CODEC_COPY]
VALID_AUDIO_CODECS = [AUDIO_CODEC_OPUS, VIDEO_CODEC_COPY]
BASIC_INFO_SCHEMA = vol.Schema({vol.Optional(CONF_NAME): cv.string, vol.Optional(CONF_LINKED_BATTERY_SENSOR): cv.entity_domain(sensor.DOMAIN), vol.Optional(CONF_LINKED_BATTERY_CHARGING_SENSOR): cv.entity_domain(binary_sensor.DOMAIN), vol.Optional(CONF_LOW_BATTERY_THRESHOLD, default=DEFAULT_LOW_BATTERY_THRESHOLD): cv.positive_int})
FEATURE_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_FEATURE_LIST, default=None): cv.ensure_list})
CAMERA_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_STREAM_ADDRESS): vol.All(ipaddress.ip_address, cv.string), vol.Optional(CONF_STREAM_SOURCE): cv.string, vol.Optional(CONF_AUDIO_CODEC, default=DEFAULT_AUDIO_CODEC): vol.In(VALID_AUDIO_CODECS), vol.Optional(CONF_SUPPORT_AUDIO, default=DEFAULT_SUPPORT_AUDIO): cv.boolean, vol.Optional(CONF_MAX_WIDTH, default=DEFAULT_MAX_WIDTH): cv.positive_int, vol.Optional(CONF_MAX_HEIGHT, default=DEFAULT_MAX_HEIGHT): cv.positive_int, vol.Optional(CONF_MAX_FPS, default=DEFAULT_MAX_FPS): cv.positive_int, vol.Optional(CONF_AUDIO_MAP, default=DEFAULT_AUDIO_MAP): cv.string, vol.Optional(CONF_VIDEO_MAP, default=DEFAULT_VIDEO_MAP): cv.string, vol.Optional(CONF_STREAM_COUNT, default=DEFAULT_STREAM_COUNT): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)), vol.Optional(CONF_VIDEO_CODEC, default=DEFAULT_VIDEO_CODEC): vol.In(VALID_VIDEO_CODECS), vol.Optional(CONF_VIDEO_PROFILE_NAMES, default=DEFAULT_VIDEO_PROFILE_NAMES): [cv.string], vol.Optional(CONF_AUDIO_PACKET_SIZE, default=DEFAULT_AUDIO_PACKET_SIZE): cv.positive_int, vol.Optional(CONF_VIDEO_PACKET_SIZE, default=DEFAULT_VIDEO_PACKET_SIZE): cv.positive_int, vol.Optional(CONF_LINKED_MOTION_SENSOR): cv.entity_domain([binary_sensor.DOMAIN, EVENT_DOMAIN]), vol.Optional(CONF_LINKED_DOORBELL_SENSOR): cv.entity_domain([binary_sensor.DOMAIN, EVENT_DOMAIN])})
HUMIDIFIER_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_LINKED_HUMIDITY_SENSOR): cv.entity_domain(sensor.DOMAIN)})
COVER_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_LINKED_OBSTRUCTION_SENSOR): cv.entity_domain(binary_sensor.DOMAIN)})
CODE_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(ATTR_CODE, default=None): vol.Any(None, cv.string)})
LOCK_SCHEMA = CODE_SCHEMA.extend({vol.Optional(CONF_LINKED_DOORBELL_SENSOR): cv.entity_domain([binary_sensor.DOMAIN, EVENT_DOMAIN])})
MEDIA_PLAYER_SCHEMA = vol.Schema({vol.Required(CONF_FEATURE): vol.All(cv.string, vol.In((FEATURE_ON_OFF, FEATURE_PLAY_PAUSE, FEATURE_PLAY_STOP, FEATURE_TOGGLE_MUTE)))})
SWITCH_TYPE_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_TYPE, default=TYPE_SWITCH): vol.All(cv.string, vol.In((TYPE_FAUCET, TYPE_OUTLET, TYPE_SHOWER, TYPE_SPRINKLER, TYPE_SWITCH, TYPE_VALVE)))})
SENSOR_SCHEMA = BASIC_INFO_SCHEMA.extend({vol.Optional(CONF_THRESHOLD_CO): vol.Any(None, cv.positive_int), vol.Optional(CONF_THRESHOLD_CO2): vol.Any(None, cv.positive_int)})
HOMEKIT_CHAR_TRANSLATIONS = {0: ' ', 10: ' ', 13: ' ', 33: '-', 34: ' ', 36: '-', 37: '-', 40: '-', 41: '-', 42: '-', 43: '-', 47: '-', 58: '-', 59: '-', 60: '-', 61: '-', 62: '-', 63: '-', 64: '-', 91: '-', 92: '-', 93: '-', 94: '-', 95: ' ', 96: '-', 123: '-', 124: '-', 125: '-', 126: '-', 127: '-'}

def validate_entity_config(values: dict) -> dict:
    ...

def get_media_player_features(state: State) -> list[str]:
    ...

def validate_media_player_features(state: State, feature_list: list[str]) -> bool:
    ...

def async_show_setup_message(hass: HomeAssistant, entry_id: str, bridge_name: str, pincode: bytes, uri: str) -> None:
    ...

def async_dismiss_setup_message(hass: HomeAssistant, entry_id: str) -> None:
    ...

def convert_to_float(state: Any) -> float:
    ...

def coerce_int(state: Any) -> int:
    ...

def cleanup_name_for_homekit(name: str) -> str:
    ...

def temperature_to_homekit(temperature: float, unit: str) -> float:
    ...

def temperature_to_states(temperature: float, unit: str) -> float:
    ...

def density_to_air_quality(density: float) -> int:
    ...

def density_to_air_quality_pm10(density: float) -> int:
    ...

def density_to_air_quality_nitrogen_dioxide(density: float) -> int:
    ...

def density_to_air_quality_voc(density: float) -> int:
    ...

def get_persist_filename_for_entry_id(entry_id: str) -> str:
    ...

def get_aid_storage_filename_for_entry_id(entry_id: str) -> str:
    ...

def get_iid_storage_filename_for_entry_id(entry_id: str) -> str:
    ...

def get_persist_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
    ...

def get_aid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
    ...

def get_iid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
    ...

def _format_version_part(version_part: str) -> str:
    ...

def format_version(version: str) -> str:
    ...

def _is_zero_but_true(value: str) -> bool:
    ...

def remove_state_files_for_entry_id(hass: HomeAssistant, entry_id: str) -> None:
    ...

def _get_test_socket() -> socket.socket:
    ...

@callback
def async_port_is_available(port: int) -> bool:
    ...

@callback
def async_find_next_available_port(hass: HomeAssistant, start_port: int) -> int:
    ...

@callback
def _async_find_next_available_port(start_port: int, exclude_ports: set[int]) -> int:
    ...

def pid_is_alive(pid: int) -> bool:
    ...

def accessory_friendly_name(hass_name: str, accessory: Accessory) -> str:
    ...

def state_needs_accessory_mode(state: State) -> bool:
    ...

def state_changed_event_is_same_state(event: Event) -> bool:
    ...

def get_min_max(value1: Any, value2: Any) -> tuple[Any, Any]:
    ...
