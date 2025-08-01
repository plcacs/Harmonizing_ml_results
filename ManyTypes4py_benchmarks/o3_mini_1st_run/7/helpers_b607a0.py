from __future__ import annotations
from collections.abc import Callable
from dataclasses import astuple, dataclass
import logging
from typing import Any, Optional, Union, Dict, List, Set, Tuple, cast

import voluptuous as vol
from zwave_js_server.client import Client as ZwaveClient
from zwave_js_server.const import LOG_LEVEL_MAP, CommandClass, ConfigurationValueType, LogLevel
from zwave_js_server.model.controller import Controller
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.log_config import LogConfig
from zwave_js_server.model.node import Node as ZwaveNode
from zwave_js_server.model.value import ConfigurationValue, Value as ZwaveValue, ValueDataType, get_value_id_str
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import ATTR_AREA_ID, ATTR_DEVICE_ID, ATTR_ENTITY_ID, CONF_TYPE, __version__ as HA_VERSION
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.group import expand_entity_ids
from homeassistant.helpers.typing import ConfigType, VolSchemaType
from .const import ATTR_COMMAND_CLASS, ATTR_ENDPOINT, ATTR_PROPERTY, ATTR_PROPERTY_KEY, DATA_CLIENT, DATA_OLD_SERVER_LOG_LEVEL, DOMAIN, LIB_LOGGER, LOGGER


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

    def __post_init__(self) -> None:
        """Post initialization check."""
        if all(val is None for val in astuple(self)):
            raise ValueError('At least one of the fields must be set.')


def value_matches_matcher(matcher: ZwaveValueMatcher, value_data: Dict[str, Any]) -> bool:
    """Return whether value matches matcher."""
    command_class: Optional[CommandClass] = None
    if 'commandClass' in value_data:
        command_class = CommandClass(value_data['commandClass'])
    zwave_value_id = ZwaveValueMatcher(
        property_=value_data.get('property'),
        command_class=command_class,
        endpoint=value_data.get('endpoint'),
        property_key=value_data.get('propertyKey'),
    )
    return all(
        redacted_field_val is None or redacted_field_val == zwave_value_field_val
        for redacted_field_val, zwave_value_field_val in zip(astuple(matcher), astuple(zwave_value_id), strict=False)
    )


def get_value_id_from_unique_id(unique_id: str) -> Optional[str]:
    """Get the value ID and optional state key from a unique ID.

    Raises ValueError
    """
    split_unique_id = unique_id.split('.')
    if '-' in (value_id := split_unique_id[1]):
        return value_id
    return None


def get_state_key_from_unique_id(unique_id: str) -> Optional[int]:
    """Get the state key from a unique ID."""
    if len((split_unique_id := unique_id.split('.'))) > 2:
        try:
            return int(split_unique_id[-1])
        except ValueError:
            pass
    return None


def get_value_of_zwave_value(value: Optional[ZwaveValue]) -> Any:
    """Return the value of a ZwaveValue."""
    return value.value if value else None


async def async_enable_statistics(driver: Driver) -> None:
    """Enable statistics on the driver."""
    await driver.async_enable_statistics('Home Assistant', HA_VERSION)


async def async_enable_server_logging_if_needed(
    hass: HomeAssistant, entry: ConfigEntry, driver: Optional[Driver]
) -> None:
    """Enable logging of zwave-js-server in the lib."""
    if not driver or not driver.client.connected or driver.client.server_logging_enabled:
        return
    LOGGER.info('Enabling zwave-js-server logging')
    if (curr_server_log_level := driver.log_config.level) and LOG_LEVEL_MAP[curr_server_log_level] > LIB_LOGGER.getEffectiveLevel():
        entry_data: Dict[str, Any] = entry.runtime_data
        entry_data[DATA_OLD_SERVER_LOG_LEVEL] = curr_server_log_level
        await driver.async_update_log_config(LogConfig(level=LogLevel.DEBUG))
    await driver.client.enable_server_logging()
    LOGGER.info('Zwave-js-server logging is enabled')


async def async_disable_server_logging_if_needed(
    hass: HomeAssistant, entry: ConfigEntry, driver: Optional[Driver]
) -> None:
    """Disable logging of zwave-js-server in the lib if still connected to server."""
    if not driver or not driver.client.connected or (not driver.client.server_logging_enabled):
        return
    LOGGER.info('Disabling zwave_js server logging')
    if DATA_OLD_SERVER_LOG_LEVEL in entry.runtime_data and (
        old_server_log_level := entry.runtime_data.pop(DATA_OLD_SERVER_LOG_LEVEL)
    ) != driver.log_config.level:
        LOGGER.info(
            'Server logging is currently set to %s as a result of server logging being enabled. It is now being reset to %s',
            driver.log_config.level,
            old_server_log_level,
        )
        await driver.async_update_log_config(LogConfig(level=old_server_log_level))
    await driver.client.disable_server_logging()
    LOGGER.info('Zwave-js-server logging is enabled')


def get_valueless_base_unique_id(driver: Driver, node: ZwaveNode) -> str:
    """Return the base unique ID for an entity that is not based on a value."""
    return f'{driver.controller.home_id}.{node.node_id}'


def get_unique_id(driver: Driver, value_id: Any) -> str:
    """Get unique ID from client and value ID."""
    return f'{driver.controller.home_id}.{value_id}'


def get_device_id(driver: Driver, node: ZwaveNode) -> Tuple[str, str]:
    """Get device registry identifier for Z-Wave node."""
    return (DOMAIN, f'{driver.controller.home_id}-{node.node_id}')


def get_device_id_ext(driver: Driver, node: ZwaveNode) -> Optional[Tuple[str, str]]:
    """Get extended device registry identifier for Z-Wave node."""
    if None in (node.manufacturer_id, node.product_type, node.product_id):
        return None
    domain, dev_id = get_device_id(driver, node)
    return (domain, f'{dev_id}-{node.manufacturer_id}:{node.product_type}:{node.product_id}')


def get_home_and_node_id_from_device_entry(
    device_entry: dr.DeviceEntry
) -> Optional[Tuple[str, int]]:
    """Get home ID and node ID for Z-Wave device registry entry.

    Returns (home_id, node_id) or None if not found.
    """
    device_id = next((identifier[1] for identifier in device_entry.identifiers if identifier[0] == DOMAIN), None)
    if device_id is None:
        return None
    id_ = device_id.split('-')
    return (id_[0], int(id_[1]))


@callback
def async_get_node_from_device_id(
    hass: HomeAssistant, device_id: str, dev_reg: Optional[dr.DeviceRegistry] = None
) -> ZwaveNode:
    """Get node from a device ID.

    Raises ValueError if device is invalid or node can't be found.
    """
    if not dev_reg:
        dev_reg = dr.async_get(hass)
    if not (device_entry := dev_reg.async_get(device_id)):
        raise ValueError(f'Device ID {device_id} is not valid')
    config_entry_ids: List[str] = device_entry.config_entries
    entry = next(
        (entry for entry in hass.config_entries.async_entries(DOMAIN) if entry.entry_id in config_entry_ids), None
    )
    if entry and entry.state != ConfigEntryState.LOADED:
        raise ValueError(f'Device {device_id} config entry is not loaded')
    if entry is None:
        raise ValueError(f'Device {device_id} is not from an existing zwave_js config entry')
    client: ZwaveClient = entry.runtime_data[DATA_CLIENT]
    driver: Optional[Driver] = client.driver
    if driver is None:
        raise ValueError('Driver is not ready.')
    identifiers = get_home_and_node_id_from_device_entry(device_entry)
    node_id: Optional[int] = identifiers[1] if identifiers else None
    if node_id is None or node_id not in driver.controller.nodes:
        raise ValueError(f"Node for device {device_id} can't be found")
    return driver.controller.nodes[node_id]


@callback
def async_get_node_from_entity_id(
    hass: HomeAssistant,
    entity_id: str,
    ent_reg: Optional[er.EntityRegistry] = None,
    dev_reg: Optional[dr.DeviceRegistry] = None,
) -> ZwaveNode:
    """Get node from an entity ID.

    Raises ValueError if entity is invalid.
    """
    if not ent_reg:
        ent_reg = er.async_get(hass)
    entity_entry = ent_reg.async_get(entity_id)
    if entity_entry is None or entity_entry.platform != DOMAIN:
        raise ValueError(f'Entity {entity_id} is not a valid {DOMAIN} entity')
    assert entity_entry.device_id
    return async_get_node_from_device_id(hass, entity_entry.device_id, dev_reg)


@callback
def async_get_nodes_from_area_id(
    hass: HomeAssistant,
    area_id: str,
    ent_reg: Optional[er.EntityRegistry] = None,
    dev_reg: Optional[dr.DeviceRegistry] = None,
) -> Set[ZwaveNode]:
    """Get nodes for all Z-Wave JS devices and entities that are in an area."""
    nodes: Set[ZwaveNode] = set()
    if ent_reg is None:
        ent_reg = er.async_get(hass)
    if dev_reg is None:
        dev_reg = dr.async_get(hass)
    nodes.update({
        async_get_node_from_device_id(hass, entity.device_id, dev_reg)
        for entity in er.async_entries_for_area(ent_reg, area_id)
        if entity.platform == DOMAIN and entity.device_id is not None
    })
    nodes.update((
        async_get_node_from_device_id(hass, device.id, dev_reg)
        for device in dr.async_entries_for_area(dev_reg, area_id)
        if any(
            (
                cast(ConfigEntry, hass.config_entries.async_get_entry(config_entry_id)).domain == DOMAIN
                for config_entry_id in device.config_entries
            )
        )
    ))
    return nodes


@callback
def async_get_nodes_from_targets(
    hass: HomeAssistant,
    val: Dict[str, Any],
    ent_reg: Optional[er.EntityRegistry] = None,
    dev_reg: Optional[dr.DeviceRegistry] = None,
    logger: logging.Logger = LOGGER,
) -> Set[ZwaveNode]:
    """Get nodes for all targets.

    Supports entity_id with group expansion, area_id, and device_id.
    """
    nodes: Set[ZwaveNode] = set()
    for entity_id in expand_entity_ids(hass, val.get(ATTR_ENTITY_ID, [])):
        try:
            nodes.add(async_get_node_from_entity_id(hass, entity_id, ent_reg, dev_reg))
        except ValueError as err:
            logger.warning(err.args[0])
    for area_id in val.get(ATTR_AREA_ID, []):
        nodes.update(async_get_nodes_from_area_id(hass, area_id, ent_reg, dev_reg))
    for device_id in val.get(ATTR_DEVICE_ID, []):
        try:
            nodes.add(async_get_node_from_device_id(hass, device_id, dev_reg))
        except ValueError as err:
            logger.warning(err.args[0])
    return nodes


def get_zwave_value_from_config(node: ZwaveNode, config: Dict[str, Any]) -> ZwaveValue:
    """Get a Z-Wave JS Value from a config."""
    endpoint: Optional[Any] = None
    if config.get(ATTR_ENDPOINT):
        endpoint = config[ATTR_ENDPOINT]
    property_key: Optional[Any] = None
    if config.get(ATTR_PROPERTY_KEY):
        property_key = config[ATTR_PROPERTY_KEY]
    value_id: str = get_value_id_str(node, config[ATTR_COMMAND_CLASS], config[ATTR_PROPERTY], endpoint, property_key)
    if value_id not in node.values:
        raise vol.Invalid(f"Value {value_id} can't be found on node {node}")
    return node.values[value_id]


def _zwave_js_config_entry(hass: HomeAssistant, device: dr.DeviceEntry) -> Optional[str]:
    """Find zwave_js config entry from a device."""
    for entry_id in device.config_entries:
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry and entry.domain == DOMAIN:
            return entry_id
    return None


@callback
def async_get_node_status_sensor_entity_id(
    hass: HomeAssistant,
    device_id: str,
    ent_reg: Optional[er.EntityRegistry] = None,
    dev_reg: Optional[dr.DeviceRegistry] = None,
) -> Optional[str]:
    """Get the node status sensor entity ID for a given Z-Wave JS device."""
    if not ent_reg:
        ent_reg = er.async_get(hass)
    if not dev_reg:
        dev_reg = dr.async_get(hass)
    if not (device := dev_reg.async_get(device_id)):
        raise HomeAssistantError('Invalid Device ID provided')
    if not (entry_id := _zwave_js_config_entry(hass, device)):
        return None
    entry: ConfigEntry = hass.config_entries.async_get_entry(entry_id)
    assert entry
    client: ZwaveClient = entry.runtime_data[DATA_CLIENT]
    node: ZwaveNode = async_get_node_from_device_id(hass, device_id, dev_reg)
    return ent_reg.async_get_entity_id(
        SENSOR_DOMAIN, DOMAIN, f'{client.driver.controller.home_id}.{node.node_id}.node_status'
    )


def remove_keys_with_empty_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys from config where the value is an empty string or None."""
    return {key: value for key, value in config.items() if value not in ('', None)}


def check_type_schema_map(
    schema_map: Dict[str, Callable[[ConfigType], ConfigType]]
) -> Callable[[ConfigType], ConfigType]:
    """Check type specific schema against config."""

    def _check_type_schema(config: ConfigType) -> ConfigType:
        """Check type specific schema against config."""
        return cast(ConfigType, schema_map[str(config[CONF_TYPE])](config))

    return _check_type_schema


def copy_available_params(input_dict: Dict[str, Any], output_dict: Dict[str, Any], params: List[str]) -> None:
    """Copy available params from input into output."""
    output_dict.update({param: input_dict[param] for param in params if param in input_dict})


def get_value_state_schema(value: Any) -> Any:
    """Return device automation schema for a config entry."""
    if isinstance(value, ConfigurationValue):
        min_ = value.metadata.min
        max_ = value.metadata.max
        if value.configuration_value_type in (ConfigurationValueType.RANGE, ConfigurationValueType.MANUAL_ENTRY):
            return vol.All(vol.Coerce(int), vol.Range(min=min_, max=max_))
        if value.configuration_value_type == ConfigurationValueType.BOOLEAN:
            return vol.Coerce(bool)
        if value.configuration_value_type == ConfigurationValueType.ENUMERATED:
            return vol.In({int(k): v for k, v in value.metadata.states.items()})
        return None
    if value.metadata.states:
        return vol.In({int(k): v for k, v in value.metadata.states.items()})
    return vol.All(vol.Coerce(int), vol.Range(min=value.metadata.min, max=value.metadata.max))


def get_device_info(driver: Driver, node: ZwaveNode) -> DeviceInfo:
    """Get DeviceInfo for node."""
    return DeviceInfo(
        identifiers={get_device_id(driver, node)},
        sw_version=node.firmware_version,
        name=node.name or node.device_config.description or f'Node {node.node_id}',
        model=node.device_config.label,
        manufacturer=node.device_config.manufacturer,
        suggested_area=node.location if node.location else None,
    )


def get_network_identifier_for_notification(
    hass: HomeAssistant, config_entry: ConfigEntry, controller: Controller
) -> str:
    """Return the network identifier string for persistent notifications."""
    home_id: str = str(controller.home_id)
    if len(hass.config_entries.async_entries(DOMAIN)) > 1:
        if str(home_id) != config_entry.title:
            return f'`{config_entry.title}`, with the home ID `{home_id}`,'
        return f'with the home ID `{home_id}`'
    return ''
