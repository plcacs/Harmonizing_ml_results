"""Helper functions for Z-Wave JS integration."""
from collections.abc import Callable
from typing import Any, Optional, Union, Sequence, Iterable, Set, Tuple, Dict
import voluptuous as vol
from zwave_js_server.const import CommandClass, LogLevel
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.log_config import LogConfig
from zwave_js_server.model.node import Node as ZwaveNode
from zwave_js_server.model.value import ConfigurationValue, Value as ZwaveValue
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceInfo, DeviceRegistry
from homeassistant.helpers.entity_registry import EntityRegistry

class ZwaveValueID:
    """Class to represent a value ID."""
    endpoint: Optional[int]
    property_key: Optional[str]

class ZwaveValueMatcher:
    """Class to allow matching a Z-Wave Value."""
    property_: Optional[str]
    command_class: Optional[CommandClass]
    endpoint: Optional[int]
    property_key: Optional[str]

    def __post_init__(self) -> None: ...

def value_matches_matcher(matcher: ZwaveValueMatcher, value_data: Dict[str, Any]) -> bool:
    """Return whether value matches matcher."""
    ...

def get_value_id_from_unique_id(unique_id: str) -> Optional[str]:
    """Get the value ID and optional state key from a unique ID.

    Raises ValueError
    """
    ...

def get_state_key_from_unique_id(unique_id: str) -> Optional[int]:
    """Get the state key from a unique ID."""
    ...

def get_value_of_zwave_value(value: Optional[ZwaveValue]) -> Any:
    """Return the value of a ZwaveValue."""
    ...

async def async_enable_statistics(driver: Driver) -> None:
    """Enable statistics on the driver."""
    ...

async def async_enable_server_logging_if_needed(hass: HomeAssistant, entry: ConfigEntry, driver: Optional[Driver]) -> None:
    """Enable logging of zwave-js-server in the lib."""
    ...

async def async_disable_server_logging_if_needed(hass: HomeAssistant, entry: ConfigEntry, driver: Optional[Driver]) -> None:
    """Disable logging of zwave-js-server in the lib if still connected to server."""
    ...

def get_valueless_base_unique_id(driver: Driver, node: ZwaveNode) -> str:
    """Return the base unique ID for an entity that is not based on a value."""
    ...

def get_unique_id(driver: Driver, value_id: str) -> str:
    """Get unique ID from client and value ID."""
    ...

def get_device_id(driver: Driver, node: ZwaveNode) -> Tuple[str, str]:
    """Get device registry identifier for Z-Wave node."""
    ...

def get_device_id_ext(driver: Driver, node: ZwaveNode) -> Optional[Tuple[str, str]]:
    """Get extended device registry identifier for Z-Wave node."""
    ...

def get_home_and_node_id_from_device_entry(device_entry: Any) -> Optional[Tuple[str, int]]:
    """Get home ID and node ID for Z-Wave device registry entry.

    Returns (home_id, node_id) or None if not found.
    """
    ...

def async_get_node_from_device_id(
    hass: HomeAssistant, 
    device_id: str, 
    dev_reg: Optional[DeviceRegistry] = None
) -> ZwaveNode:
    """Get node from a device ID.

    Raises ValueError if device is invalid or node can't be found.
    """
    ...

def async_get_node_from_entity_id(
    hass: HomeAssistant, 
    entity_id: str, 
    ent_reg: Optional[EntityRegistry] = None, 
    dev_reg: Optional[DeviceRegistry] = None
) -> ZwaveNode:
    """Get node from an entity ID.

    Raises ValueError if entity is invalid.
    """
    ...

def async_get_nodes_from_area_id(
    hass: HomeAssistant, 
    area_id: str, 
    ent_reg: Optional[EntityRegistry] = None, 
    dev_reg: Optional[DeviceRegistry] = None
) -> Set[ZwaveNode]:
    """Get nodes for all Z-Wave JS devices and entities that are in an area."""
    ...

def async_get_nodes_from_targets(
    hass: HomeAssistant, 
    val: Dict[str, Any], 
    ent_reg: Optional[EntityRegistry] = None, 
    dev_reg: Optional[DeviceRegistry] = None, 
    logger: Any = ...
) -> Set[ZwaveNode]:
    """Get nodes for all targets.

    Supports entity_id with group expansion, area_id, and device_id.
    """
    ...

def get_zwave_value_from_config(node: ZwaveNode, config: Dict[str, Any]) -> ZwaveValue:
    """Get a Z-Wave JS Value from a config."""
    ...

def _zwave_js_config_entry(hass: HomeAssistant, device: Any) -> Optional[str]:
    """Find zwave_js config entry from a device."""
    ...

def async_get_node_status_sensor_entity_id(
    hass: HomeAssistant, 
    device_id: str, 
    ent_reg: Optional[EntityRegistry] = None, 
    dev_reg: Optional[DeviceRegistry] = None
) -> Optional[str]:
    """Get the node status sensor entity ID for a given Z-Wave JS device."""
    ...

def remove_keys_with_empty_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys from config where the value is an empty string or None."""
    ...

def check_type_schema_map(schema_map: Dict[str, Callable[[Dict[str, Any]], Any]]) -> Callable[[Dict[str, Any]], Any]:
    """Check type specific schema against config."""
    ...

def copy_available_params(input_dict: Dict[str, Any], output_dict: Dict[str, Any], params: Iterable[str]) -> None:
    """Copy available params from input into output."""
    ...

def get_value_state_schema(value: Union[ConfigurationValue, ZwaveValue]) -> Optional[vol.Schema]:
    """Return device automation schema for a config entry."""
    ...

def get_device_info(driver: Driver, node: ZwaveNode) -> DeviceInfo:
    """Get DeviceInfo for node."""
    ...

def get_network_identifier_for_notification(
    hass: HomeAssistant, 
    config_entry: ConfigEntry, 
    controller: Any
) -> str:
    """Return the network identifier string for persistent notifications."""
    ...