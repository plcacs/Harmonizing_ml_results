from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable
from enum import IntEnum
import logging
from typing import Any, DefaultDict, Dict, List, Set, Union, cast
from mysensors import BaseAsyncGateway, Message
from mysensors.sensor import ChildSensor
import voluptuous as vol
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util.decorator import Registry
from .const import (
    ATTR_DEVICES,
    ATTR_GATEWAY_ID,
    ATTR_NODE_ID,
    DOMAIN,
    FLAT_PLATFORM_TYPES,
    MYSENSORS_DISCOVERED_NODES,
    MYSENSORS_DISCOVERY,
    MYSENSORS_NODE_DISCOVERY,
    MYSENSORS_ON_UNLOAD,
    TYPE_TO_PLATFORMS,
    DevId,
    GatewayId,
    SensorType,
    ValueType,
)

_LOGGER = logging.getLogger(__name__)
SCHEMAS: Registry[tuple[str, str], Callable[[BaseAsyncGateway, ChildSensor, str], vol.Schema]] = Registry()


@callback
def on_unload(hass: HomeAssistant, gateway_id: GatewayId, fnct: Callable[[], None]) -> None:
    """Register a callback to be called when entry is unloaded.

    This function is used by platforms to cleanup after themselves.
    """
    key: str = MYSENSORS_ON_UNLOAD.format(gateway_id)
    if key not in hass.data[DOMAIN]:
        hass.data[DOMAIN][key] = []
    hass.data[DOMAIN][key].append(fnct)


@callback
def discover_mysensors_platform(
    hass: HomeAssistant, gateway_id: GatewayId, platform: str, new_devices: List[Any]
) -> None:
    """Discover a MySensors platform."""
    _LOGGER.debug("Discovering platform %s with devIds: %s", platform, new_devices)
    async_dispatcher_send(
        hass,
        MYSENSORS_DISCOVERY.format(gateway_id, platform),
        {ATTR_DEVICES: new_devices, CONF_NAME: DOMAIN, ATTR_GATEWAY_ID: gateway_id},
    )


@callback
def discover_mysensors_node(hass: HomeAssistant, gateway_id: GatewayId, node_id: int) -> None:
    """Discover a MySensors node."""
    discovered_nodes: Set[int] = hass.data[DOMAIN].setdefault(MYSENSORS_DISCOVERED_NODES.format(gateway_id), set())
    if node_id not in discovered_nodes:
        discovered_nodes.add(node_id)
        async_dispatcher_send(
            hass,
            MYSENSORS_NODE_DISCOVERY,
            {ATTR_GATEWAY_ID: gateway_id, ATTR_NODE_ID: node_id},
        )


def default_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> vol.Schema:
    """Return a default validation schema for value types."""
    schema: Dict[str, Any] = {value_type_name: cv.string}
    return get_child_schema(gateway, child, value_type_name, schema)


@SCHEMAS.register(("light", "V_DIMMER"))
def light_dimmer_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> vol.Schema:
    """Return a validation schema for V_DIMMER."""
    schema: Dict[str, Any] = {"V_DIMMER": cv.string, "V_LIGHT": cv.string}
    return get_child_schema(gateway, child, value_type_name, schema)


@SCHEMAS.register(("light", "V_PERCENTAGE"))
def light_percentage_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> vol.Schema:
    """Return a validation schema for V_PERCENTAGE."""
    schema: Dict[str, Any] = {"V_PERCENTAGE": cv.string, "V_STATUS": cv.string}
    return get_child_schema(gateway, child, value_type_name, schema)


@SCHEMAS.register(("light", "V_RGB"))
def light_rgb_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> vol.Schema:
    """Return a validation schema for V_RGB."""
    schema: Dict[str, Any] = {"V_RGB": cv.string, "V_STATUS": cv.string}
    return get_child_schema(gateway, child, value_type_name, schema)


@SCHEMAS.register(("light", "V_RGBW"))
def light_rgbw_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> vol.Schema:
    """Return a validation schema for V_RGBW."""
    schema: Dict[str, Any] = {"V_RGBW": cv.string, "V_STATUS": cv.string}
    return get_child_schema(gateway, child, value_type_name, schema)


@SCHEMAS.register(("switch", "V_IR_SEND"))
def switch_ir_send_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> vol.Schema:
    """Return a validation schema for V_IR_SEND."""
    schema: Dict[str, Any] = {"V_IR_SEND": cv.string, "V_LIGHT": cv.string}
    return get_child_schema(gateway, child, value_type_name, schema)


def get_child_schema(
    gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str, schema: Dict[str, Any]
) -> vol.Schema:
    """Return a child schema."""
    set_req = gateway.const.SetReq
    child_schema: vol.Schema = cast(vol.Schema, child.get_schema(gateway.protocol_version))
    return child_schema.extend(
        {
            vol.Required(
                set_req[name].value, msg=invalid_msg(gateway, child, name)
            ): child_schema.schema.get(set_req[name].value, cv.string)
            for name, valid in schema.items()
        },
        extra=vol.ALLOW_EXTRA,
    )


def invalid_msg(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> str:
    """Return a message for an invalid child during schema validation."""
    presentation = gateway.const.Presentation
    set_req = gateway.const.SetReq
    return f"{presentation(child.type).name} requires value_type {set_req[value_type_name].name}"


def validate_set_msg(gateway_id: GatewayId, msg: Message) -> Dict[Any, Any]:
    """Validate a set message."""
    if not validate_node(msg.gateway, msg.node_id):
        return {}
    child: ChildSensor = msg.gateway.sensors[msg.node_id].children[msg.child_id]
    return validate_child(gateway_id, msg.gateway, msg.node_id, child, msg.sub_type)


def validate_node(gateway: BaseAsyncGateway, node_id: int) -> bool:
    """Validate a node."""
    if gateway.sensors[node_id].sketch_name is None:
        _LOGGER.debug("Node %s is missing sketch name", node_id)
        return False
    return True


def validate_child(
    gateway_id: GatewayId,
    gateway: BaseAsyncGateway,
    node_id: int,
    child: ChildSensor,
    value_type: Union[ValueType, None] = None,
) -> DefaultDict[str, List[DevId]]:
    """Validate a child. Returns a dict mapping hass platform names to list of DevId."""
    validated: DefaultDict[str, List[DevId]] = defaultdict(list)
    presentation = gateway.const.Presentation
    set_req = gateway.const.SetReq
    child_type_name: Union[str, None] = next(
        (member.name for member in presentation if member.value == child.type), None
    )
    if not child_type_name:
        _LOGGER.warning("Child type %s is not supported", child.type)
        return validated
    value_types: Set[ValueType] = {value_type} if value_type is not None else {*(child.values)}
    value_type_names: Set[str] = {member.name for member in set_req if member.value in value_types}
    platforms: List[str] = TYPE_TO_PLATFORMS.get(child_type_name, [])
    if not platforms:
        _LOGGER.warning("Child type %s is not supported", child.type)
        return validated
    for platform in platforms:
        platform_v_names: Set[str] = FLAT_PLATFORM_TYPES[platform, child_type_name]
        v_names: Set[str] = platform_v_names & value_type_names
        if not v_names:
            child_value_names: Set[str] = {member.name for member in set_req if member.value in child.values}
            v_names = platform_v_names & child_value_names
        for v_name in v_names:
            child_schema_gen: Callable[[BaseAsyncGateway, ChildSensor, str], vol.Schema] = SCHEMAS.get(
                (platform, v_name), default_schema
            )
            child_schema: vol.Schema = child_schema_gen(gateway, child, v_name)
            try:
                child_schema(child.values)
            except vol.Invalid as exc:
                _LOGGER.warning("Invalid %s on node %s, %s platform: %s", child, node_id, platform, exc)
                continue
            dev_id: DevId = (gateway_id, node_id, child.id, set_req[v_name].value)
            validated[platform].append(dev_id)
    return validated