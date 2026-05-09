"""Stub file for homeassistant.components.homekit.util_e9973a"""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
    overload,
)
from pyhap.accessory import Accessory
from homeassistant.components import (
    binary_sensor,
    media_player,
    persistent_notification,
    sensor,
)
from homeassistant.const import (
    ATTR_CODE,
    ATTR_DEVICE_CLASS,
    ATTR_SUPPORTED_FEATURES,
    CONF_NAME,
    CONF_PORT,
    CONF_TYPE,
    UnitOfTemperature,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers.storage import STORAGE_DIR
from homeassistant.util.unit_conversion import TemperatureConverter
import voluptuous as vol
import io
import ipaddress
import logging
import os
import re
import secrets
import socket

HOMEKIT_CHAR_TRANSLATIONS: dict[int, str] = ...
INVALID_END_CHARS: str = ...
MAX_VERSION_PART: int = ...
MAX_PORT: int = ...
VALID_VIDEO_CODECS: list[str] = ...
VALID_AUDIO_CODECS: list[str] = ...
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

def validate_entity_config(values: dict) -> dict[str, dict[str, Any]]:
    ...

def get_media_player_features(state: State) -> list[str]:
    ...

def validate_media_player_features(state: State, feature_list: list[str]) -> bool:
    ...

@callback
def async_show_setup_message(
    hass: HomeAssistant,
    entry_id: str,
    bridge_name: str,
    pincode: bytes,
    uri: str,
) -> None:
    ...

@callback
def async_dismiss_setup_message(hass: HomeAssistant, entry_id: str) -> None:
    ...

def convert_to_float(state: Any) -> Optional[float]:
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

def format_version(version: Union[str, int]) -> Optional[str]:
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
def _async_find_next_available_port(start_port: int, exclude_ports: Set[int]) -> int:
    ...

def pid_is_alive(pid: int) -> bool:
    ...

def accessory_friendly_name(hass_name: str, accessory: Accessory) -> str:
    ...

def state_needs_accessory_mode(state: State) -> bool:
    ...

def state_changed_event_is_same_state(event: Event) -> bool:
    ...

def get_min_max(value1: Union[int, float], value2: Union[int, float]) -> Tuple[Union[int, float], Union[int, float]]:
    ...