from __future__ import annotations
import io
import ipaddress
import logging
import os
import re
import secrets
import socket
from typing import Any, Dict, List, Tuple, Union, cast
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
from homeassistant.core import Event, HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.storage import STORAGE_DIR
from homeassistant.util.unit_conversion import TemperatureConverter
from .const import AUDIO_CODEC_COPY, AUDIO_CODEC_OPUS, CONF_AUDIO_CODEC, CONF_AUDIO_MAP, CONF_AUDIO_PACKET_SIZE, CONF_FEATURE, CONF_FEATURE_LIST, CONF_LINKED_BATTERY_CHARGING_SENSOR, CONF_LINKED_BATTERY_SENSOR, CONF_LINKED_DOORBELL_SENSOR, CONF_LINKED_HUMIDITY_SENSOR, CONF_LINKED_MOTION_SENSOR, CONF_LINKED_OBSTRUCTION_SENSOR, CONF_LOW_BATTERY_THRESHOLD, CONF_MAX_FPS, CONF_MAX_HEIGHT, CONF_MAX_WIDTH, CONF_STREAM_ADDRESS, CONF_STREAM_COUNT, CONF_STREAM_SOURCE, CONF_SUPPORT_AUDIO, CONF_THRESHOLD_CO, CONF_THRESHOLD_CO2, CONF_VIDEO_CODEC, CONF_VIDEO_MAP, CONF_VIDEO_PACKET_SIZE, CONF_VIDEO_PROFILE_NAMES, DEFAULT_AUDIO_CODEC, DEFAULT_AUDIO_MAP, DEFAULT_AUDIO_PACKET_SIZE, DEFAULT_LOW_BATTERY_THRESHOLD, DEFAULT_MAX_FPS, DEFAULT_MAX_HEIGHT, DEFAULT_MAX_WIDTH, DEFAULT_STREAM_COUNT, DEFAULT_SUPPORT_AUDIO, DEFAULT_VIDEO_CODEC, DEFAULT_VIDEO_MAP, DEFAULT_VIDEO_PACKET_SIZE, DEFAULT_VIDEO_PROFILE_NAMES, DOMAIN, FEATURE_ON_OFF, FEATURE_PLAY_PAUSE, FEATURE_PLAY_STOP, FEATURE_TOGGLE_MUTE, MAX_NAME_LENGTH, TYPE_FAUCET, TYPE_OUTLET, TYPE_SHOWER, TYPE_SPRINKLER, TYPE_SWITCH, TYPE_VALVE, VIDEO_CODEC_COPY, VIDEO_CODEC_H264_OMX, VIDEO_CODEC_H264_V4L2M2M, VIDEO_CODEC_LIBX264
from .models import HomeKitConfigEntry

_LOGGER: logging.Logger

NUMBERS_ONLY_RE: re.Pattern
VERSION_RE: re.Pattern
INVALID_END_CHARS: str
MAX_VERSION_PART: int
MAX_PORT: int
VALID_VIDEO_CODECS: List[str]
VALID_AUDIO_CODECS: List[str]
HOMEKIT_CHAR_TRANSLATIONS: Dict[int, str]

def validate_entity_config(values: Any) -> Dict[str, Any]:
def get_media_player_features(state: State) -> List[str]:
def validate_media_player_features(state: State, feature_list: List[str]) -> bool:
def async_show_setup_message(hass: HomeAssistant, entry_id: str, bridge_name: str, pincode: bytes, uri: str) -> None:
def async_dismiss_setup_message(hass: HomeAssistant, entry_id: str) -> None:
def convert_to_float(state: Any) -> Union[float, None]:
def coerce_int(state: Any) -> int:
def cleanup_name_for_homekit(name: str) -> str:
def temperature_to_homekit(temperature: float, unit: str) -> float:
def temperature_to_states(temperature: float, unit: str) -> float:
def density_to_air_quality(density: float) -> int:
def density_to_air_quality_pm10(density: float) -> int:
def density_to_air_quality_nitrogen_dioxide(density: float) -> int:
def density_to_air_quality_voc(density: float) -> int:
def get_persist_filename_for_entry_id(entry_id: str) -> str:
def get_aid_storage_filename_for_entry_id(entry_id: str) -> str:
def get_iid_storage_filename_for_entry_id(entry_id: str) -> str:
def get_persist_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
def get_aid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
def get_iid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str:
def _format_version_part(version_part: str) -> str:
def format_version(version: str) -> Union[str, None]:
def _is_zero_but_true(value: str) -> bool:
def remove_state_files_for_entry_id(hass: HomeAssistant, entry_id: str) -> None:
def _get_test_socket() -> socket.socket:
def async_port_is_available(port: int) -> bool:
def async_find_next_available_port(hass: HomeAssistant, start_port: int) -> int:
def _async_find_next_available_port(start_port: int, exclude_ports: set) -> int:
def pid_is_alive(pid: int) -> bool:
def accessory_friendly_name(hass_name: str, accessory: Accessory) -> str:
def state_needs_accessory_mode(state: State) -> bool:
def state_changed_event_is_same_state(event: Event) -> bool:
def get_min_max(value1: Union[int, float], value2: Union[int, float]) -> Tuple[Union[int, float], Union[int, float]]:
