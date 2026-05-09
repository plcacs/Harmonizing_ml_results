"""Stub file for 'util_e9973a' module."""

from __future__ import annotations
from collections.abc import Iterable
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
)
from pyhap.accessory import Accessory
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.const import UnitOfTemperature
from homeassistant.components import (
    binary_sensor,
    media_player,
    persistent_notification,
    sensor,
)
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.event import DOMAIN as EVENT_DOMAIN
from homeassistant.components.lock import DOMAIN as LOCK_DOMAIN
from homeassistant.components.media_player import (
    DOMAIN as MEDIA_PLAYER_DOMAIN,
    MediaPlayerDeviceClass,
    MediaPlayerEntityFeature,
)
from homeassistant.components.remote import (
    DOMAIN as REMOTE_DOMAIN,
    RemoteEntityFeature,
)

BASIC_INFO_SCHEMA: Any = ...
FEATURE_SCHEMA: Any = ...
CAMERA_SCHEMA: Any = ...
HUMIDIFIER_SCHEMA: Any = ...
COVER_SCHEMA: Any = ...
CODE_SCHEMA: Any = ...
LOCK_SCHEMA: Any = ...
MEDIA_PLAYER_SCHEMA: Any = ...
SWITCH_TYPE_SCHEMA: Any = ...
SENSOR_SCHEMA: Any = ...
HOMEKIT_CHAR_TRANSLATIONS: Dict[int, str] = ...


def validate_entity_config(values: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ...


def get_media_player_features(state: State) -> List[str]:
    ...


def validate_media_player_features(state: State, feature_list: List[str]) -> bool:
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


def convert_to_float(state: State) -> Union[float, None]:
    ...


def coerce_int(state: State) -> int:
    ...


def cleanup_name_for_homekit(name: Optional[str]) -> str:
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


def format_version(version: Any) -> Union[str, None]:
    ...


def remove_state_files_for_entry_id(hass: HomeAssistant, entry_id: str) -> None:
    ...


def async_port_is_available(port: int) -> bool:
    ...


@callback
def async_find_next_available_port(
    hass: HomeAssistant,
    start_port: int,
) -> int:
    ...


@callback
def _async_find_next_available_port(
    start_port: int,
    exclude_ports: Set[int],
) -> int:
    ...


def pid_is_alive(pid: int) -> bool:
    ...


def accessory_friendly_name(hass_name: str, accessory: Accessory) -> str:
    ...


def state_needs_accessory_mode(state: State) -> bool:
    ...


def state_changed_event_is_same_state(event: Event) -> bool:
    ...


def get_min_max(value1: float, value2: float) -> Tuple[float, float]:
    ...