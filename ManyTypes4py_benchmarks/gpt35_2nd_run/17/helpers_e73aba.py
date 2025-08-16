from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable
from enum import IntEnum
import logging
from typing import cast, Dict, Any, Set, List, Tuple
from mysensors import BaseAsyncGateway, Message
from mysensors.sensor import ChildSensor
import voluptuous as vol
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util.decorator import Registry
from .const import ATTR_DEVICES, ATTR_GATEWAY_ID, ATTR_NODE_ID, DOMAIN, FLAT_PLATFORM_TYPES, MYSENSORS_DISCOVERED_NODES, MYSENSORS_DISCOVERY, MYSENSORS_NODE_DISCOVERY, MYSENSORS_ON_UNLOAD, TYPE_TO_PLATFORMS, DevId, GatewayId, SensorType, ValueType

_LOGGER: logging.Logger = logging.getLogger(__name__)
SCHEMAS: Registry = Registry()

def on_unload(hass: HomeAssistant, gateway_id: GatewayId, fnct: Callable) -> None:
    ...

def discover_mysensors_platform(hass: HomeAssistant, gateway_id: GatewayId, platform: str, new_devices: List[DevId]) -> None:
    ...

def discover_mysensors_node(hass: HomeAssistant, gateway_id: GatewayId, node_id: int) -> None:
    ...

def default_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> Dict[str, Any]:
    ...

@SCHEMAS.register(('light', 'V_DIMMER'))
def light_dimmer_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> Dict[str, Any]:
    ...

@SCHEMAS.register(('light', 'V_PERCENTAGE'))
def light_percentage_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> Dict[str, Any]:
    ...

@SCHEMAS.register(('light', 'V_RGB'))
def light_rgb_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> Dict[str, Any]:
    ...

@SCHEMAS.register(('light', 'V_RGBW'))
def light_rgbw_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> Dict[str, Any]:
    ...

@SCHEMAS.register(('switch', 'V_IR_SEND'))
def switch_ir_send_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> Dict[str, Any]:
    ...

def get_child_schema(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str, schema: Dict[str, Any]) -> vol.Schema:
    ...

def invalid_msg(gateway: BaseAsyncGateway, child: ChildSensor, value_type_name: str) -> str:
    ...

def validate_set_msg(gateway_id: GatewayId, msg: Message) -> Dict[str, Any]:
    ...

def validate_node(gateway: BaseAsyncGateway, node_id: int) -> bool:
    ...

def validate_child(gateway_id: GatewayId, gateway: BaseAsyncGateway, node_id: int, child: ChildSensor, value_type: ValueType = None) -> Dict[str, List[DevId]]:
    ...
