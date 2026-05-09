"""Stub file for 'util_e9973a' module."""

import io
import ipaddress
import logging
import re
import socket
from typing import (
    Any,
    Awaitable,
    Dict,
    int,
    List,
    Optional,
    Tuple,
    Union,
    bool,
    str,
    bytes,
    float,
    Iterable,
    Sequence,
    Callable,
    Pattern,
    Type,
    TypeVar,
    overload,
)
import voluptuous as vol
from homeassistant.core import Event, HomeAssistant, State
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.lock import DOMAIN as LOCK_DOMAIN
from homeassistant.components.media_player import DOMAIN as MEDIA_PLAYER_DOMAIN
from homeassistant.components.remote import DOMAIN as REMOTE_DOMAIN
from pyhap.accessory import Accessory
from .const import (
    DOMAIN,
    CONF_NAME,
    CONF_LINKED_BATTERY_SENSOR,
    CONF_LINKED_BATTERY_CHARGING_SENSOR,
    CONF_LOW_BATTERY_THRESHOLD,
    CONF_LINKED_MOTION_SENSOR,
    CONF_LINKED_DOORBELL_SENSOR,
    CONF_LINKED_HUMIDITY_SENSOR,
    CONF_LINKED_OBSTRUCTION_SENSOR,
    CONF_AUDIO_CODEC,
    CONF_SUPPORT_AUDIO,
    CONF_MAX_WIDTH,
    CONF_MAX_HEIGHT,
    CONF_MAX_FPS,
    CONF_VIDEO_CODEC,
    CONF_STREAM_ADDRESS,
    CONF_STREAM_SOURCE,
    CONF_STREAM_COUNT,
    CONF_AUDIO_PACKET_SIZE,
    CONF_VIDEO_PACKET_SIZE,
    CONF_VIDEO_PROFILE_NAMES,
    CONF_TYPE,
    CONF_FEATURE,
    CONF_FEATURE_LIST,
    CONF_CODE,
    CONF_THRESHOLD_CO,
    CONF_THRESHOLD_CO2,
    DEFAULT_LOW_BATTERY_THRESHOLD,
    DEFAULT_MAX_FPS,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_WIDTH,
    DEFAULT_STREAM_COUNT,
    DEFAULT_SUPPORT_AUDIO,
    DEFAULT_AUDIO_CODEC,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_PROFILE_NAMES,
    DEFAULT_AUDIO_PACKET_SIZE,
    DEFAULT_VIDEO_PACKET_SIZE,
    DEFAULT_AUDIO_MAP,
    DEFAULT_VIDEO_MAP,
    DEFAULT_VIDEO_PACKET_SIZE,
    MAX_NAME_LENGTH,
    TYPE_FAUCET,
    TYPE_OUTLET,
    TYPE_SHOWER,
    TYPE_SPRINKLER,
    TYPE_SWITCH,
    TYPE_VALVE,
    VIDEO_CODEC_LIBX264,
    VIDEO_CODEC_H264_OMX,
    VIDEO_CODEC_H264_V4L2M2M,
    VIDEO_CODEC_COPY,
    AUDIO_CODEC_OPUS,
    AUDIO_CODEC_COPY,
)

_LOGGER: logging.Logger = ...

NUMBERS_ONLY_RE: Pattern[str] = ...
VERSION_RE: Pattern[str] = ...
INVALID_END_CHARS: str = ...
MAX_VERSION_PART: int = ...
MAX_PORT: int = ...
VALID_VIDEO_CODECS: List[str] = ...
VALID_AUDIO_CODECS: List[str] = ...
BASIC_INFO_SCHEMA: vol.Schema = ...
FEATURE_SCHEMA: vol.Schema = ...
CAMERA_SCHEMA: vol.Schema = ...
HUMIDIFIER_SCHEMA: vol.Schema = ...
COVER_SCHEMA: vol.Schema = ...
CODE_SCHEMA: vol.Schema = ...
LOCK_SCHEMA: vol.Schema = ...
MEDIA_PLAYER_SCHEMA: vol.Schema = ...
SWITCH_TYPE_SCHEMA: vol.Schema = ...
SENSOR_SCHEMA: vol.Schema = ...
HOMEKIT_CHAR_TRANSLATIONS: Dict[int, str] = ...

def validate_entity_config(values: Dict[str, Any]) -> Dict[str, Any]: ...

def get_media_player_features(state: State) -> List[str]: ...

def validate_media_player_features(state: State, feature_list: List[str]) -> bool: ...

async def async_show_setup_message(
    hass: HomeAssistant,
    entry_id: str,
    bridge_name: str,
    pincode: bytes,
    uri: str,
) -> None: ...

async def async_dismiss_setup_message(hass: HomeAssistant, entry_id: str) -> None: ...

def convert_to_float(state: State) -> Optional[float]: ...

def coerce_int(state: State) -> int: ...

def cleanup_name_for_homekit(name: str) -> str: ...

def temperature_to_homekit(temperature: float, unit: str) -> float: ...

def temperature_to_states(temperature: float, unit: str) -> float: ...

def density_to_air_quality(density: float) -> int: ...

def density_to_air_quality_pm10(density: float) -> int: ...

def density_to_air_quality_nitrogen_dioxide(density: float) -> int: ...

def density_to_air_quality_voc(density: float) -> int: ...

def get_persist_filename_for_entry_id(entry_id: str) -> str: ...

def get_aid_storage_filename_for_entry_id(entry_id: str) -> str: ...

def get_iid_storage_filename_for_entry_id(entry_id: str) -> str: ...

def get_persist_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str: ...

def get_aid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str: ...

def get_iid_storage_fullpath_for_entry_id(hass: HomeAssistant, entry_id: str) -> str: ...

def format_version(version: str) -> Optional[str]: ...

def remove_state_files_for_entry_id(hass: HomeAssistant, entry_id: str) -> None: ...

def _get_test_socket() -> socket.socket: ...

@callback
def async_port_is_available(port: int) -> Awaitable[bool]: ...

@callback
def async_find_next_available_port(
    hass: HomeAssistant,
    start_port: int,
) -> Awaitable[int]: ...

@callback
def _async_find_next_available_port(
    start_port: int,
    exclude_ports: Iterable[int],
) -> int: ...

def pid_is_alive(pid: int) -> bool: ...

def accessory_friendly_name(hass_name: str, accessory: Accessory) -> str: ...

def state_needs_accessory_mode(state: State) -> bool: ...

def state_changed_event_is_same_state(event: Event) -> bool: ...

def get_min_max(value1: float, value2: float) -> Tuple[float, float]: ...