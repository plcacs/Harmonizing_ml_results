"""Helper functions for Z-Wave JS integration."""
from __future__ import annotations
from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import voluptuous as vol
from zwave_js_server.client import Client as ZwaveClient
from zwave_js_server.const import (
    CommandClass,
    ConfigurationValueType,
    LogLevel,
    LOG_LEVEL_MAP,
)
from zwave_js_server.model.controller import Controller
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.log_config import LogConfig
from zwave_js_server.model.node import Node as ZwaveNode
from zwave_js_server.model.value import (
    ConfigurationValue,
    Value as ZwaveValue,
    ValueDataType,
)
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import ATTR_AREA_ID, ATTR_DEVICE_ID, ATTR_ENTITY_ID, CONF_TYPE
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo, DeviceRegistry
from homeassistant.helpers.entity_registry import EntityRegistry
from homeassistant.helpers.typing import ConfigType, VolSchemaType

_ConfigType = TypeVar("_ConfigType", bound=ConfigType)

@dataclass
class ZwaveValueID:
    """Class to represent a value ID."""
    endpoint: Optional[Any] = None
    property_key: Optional[Any] = None

@dataclass
class ZwaveValueMatcher:
    """Class to allow matching a Z-Wave Value."""
    property_: Optional[Any] = None
    command_class: Optional[CommandClass] = None
    endpoint: Optional[Any] = None
    property_key: Optional[Any] = None

def value_matches_matcher(matcher: ZwaveValueMatcher, value_data: dict) -> bool:
    ...

def get_value_id_from_unique_id(unique_id: str) -> Optional[str]:
    ...

def get_state_key_from_unique_id(unique_id: str) -> Optional[int]:
    ...

def get_value_of_zwave_value(value: Optional[ZwaveValue]) -> Any:
    ...

async def async_enable_statistics(driver: Driver) -> None:
    ...

async def async_enable_server_logging_if_needed(
    hass: HomeAssistant,
    entry: ConfigEntry,
    driver: Driver,
) -> None:
    ...

async def async_disable_server_logging_if_needed(
    hass: HomeAssistant,
    entry: ConfigEntry,
    driver: Driver,
) -> None:
    ...

def get_valueless_base_unique_id(driver: Driver, node: ZwaveNode) -> str:
    ...

def get_unique_id(driver: Driver, value_id: str) -> str:
    ...

def get_device_id(driver: Driver, node: ZwaveNode) -> Tuple[str, str]:
    ...

def get_device_id_ext(driver: Driver, node: ZwaveNode) -> Optional[Tuple[str, str]]:
    ...

def get_home_and_node_id_from_device_entry(
    device_entry: DeviceRegistry
) -> Optional[Tuple[str, int]]:
    ...

@callback
def async_get_node_from_device_id(
    hass: HomeAssistant,
    device_id: str,
    dev_reg: Optional[DeviceRegistry] = None,
) -> ZwaveNode:
    ...

@callback
def async_get_node_from_entity_id(
    hass: HomeAssistant,
    entity_id: str,
    ent_reg: Optional[EntityRegistry] = None,
    dev_reg: Optional[DeviceRegistry] = None,
) -> ZwaveNode:
    ...

@callback
def async_get_nodes_from_area_id(
    hass: HomeAssistant,
    area_id: str,
    ent_reg: Optional[EntityRegistry] = None,
    dev_reg: Optional[DeviceRegistry] = None,
) -> Set[ZwaveNode]:
    ...

@callback
def async_get_nodes_from_targets(
    hass: HomeAssistant,
    val: Dict,
    ent_reg: Optional[EntityRegistry] = None,
    dev_reg: Optional[DeviceRegistry] = None,
    logger: logging.Logger = LOGGER,
) -> Set[ZwaveNode]:
    ...

def get_zwave_value_from_config(
    node: ZwaveNode,
    config: Dict,
) -> ZwaveValue:
    ...

def _zwave_js_config_entry(
    hass: HomeAssistant,
    device: DeviceRegistry,
) -> Optional[str]:
    ...

@callback
def async_get_node_status_sensor_entity_id(
    hass: HomeAssistant,
    device_id: str,
    ent_reg: Optional[EntityRegistry] = None,
    dev_reg: Optional[DeviceRegistry] = None,
) -> Optional[str]:
    ...

def remove_keys_with_empty_values(config: Dict) -> Dict:
    ...

def check_type_schema_map(
    schema_map: Dict[str, Callable[[Dict], _ConfigType]]
) -> Callable[[Dict], _ConfigType]:
    ...

def copy_available_params(
    input_dict: Dict,
    output_dict: Dict,
    params: List[str],
) -> None:
    ...

def get_value_state_schema(value: Union[ZwaveValue, ConfigurationValue]) -> Optional[vol.Schema]:
    ...

def get_device_info(driver: Driver, node: ZwaveNode) -> DeviceInfo:
    ...

def get_network_identifier_for_notification(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    controller: Controller,
) -> str:
    ...