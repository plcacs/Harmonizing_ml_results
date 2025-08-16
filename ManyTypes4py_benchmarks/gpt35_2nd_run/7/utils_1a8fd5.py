from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
from ipaddress import IPv4Address, IPv6Address, ip_address
from types import MappingProxyType
from typing import Any, List, Tuple, Union, cast
from aiohttp.web import Request, WebSocketResponse
from aioshelly.block_device import COAP, Block, BlockDevice
from aioshelly.const import BLOCK_GENERATIONS, DEFAULT_COAP_PORT, DEFAULT_HTTP_PORT, MODEL_1L, MODEL_DIMMER, MODEL_DIMMER_2, MODEL_EM3, MODEL_I3, MODEL_NAMES, RPC_GENERATIONS
from aioshelly.rpc_device import RpcDevice, WsServer
from yarl import URL
from homeassistant.components import network
from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PORT, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er, issue_registry as ir, singleton
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC
from homeassistant.helpers.network import NoURLAvailableError, get_url
from homeassistant.util.dt import utcnow
from .const import API_WS_URL, BASIC_INPUTS_EVENTS_TYPES, COMPONENT_ID_PATTERN, CONF_COAP_PORT, CONF_GEN, DEVICES_WITHOUT_FIRMWARE_CHANGELOG, DOMAIN, FIRMWARE_UNSUPPORTED_ISSUE_ID, GEN1_RELEASE_URL, GEN2_BETA_RELEASE_URL, GEN2_RELEASE_URL, LOGGER, RPC_INPUTS_EVENTS_TYPES, SHBTN_INPUTS_EVENTS_TYPES, SHBTN_MODELS, SHELLY_EMIT_EVENT_PATTERN, SHIX3_1_INPUTS_EVENTS_TYPES, UPTIME_DEVIATION, VIRTUAL_COMPONENTS_MAP

@callback
def async_remove_shelly_entity(hass: HomeAssistant, domain: str, unique_id: str) -> None:
    ...

def get_number_of_channels(device: BlockDevice, block: Block) -> int:
    ...

def get_block_entity_name(device: BlockDevice, block: Block, description: str = None) -> str:
    ...

def get_block_channel_name(device: BlockDevice, block: Block) -> str:
    ...

def is_block_momentary_input(settings: dict, block: Block, include_detached: bool = False) -> bool:
    ...

def get_device_uptime(uptime: int, last_uptime: datetime) -> Union[datetime, None]:
    ...

def get_block_input_triggers(device: BlockDevice, block: Block) -> List[Tuple[str, str]]:
    ...

def get_shbtn_input_triggers() -> List[Tuple[str, str]]:
    ...

@singleton.singleton('shelly_coap')
async def get_coap_context(hass: HomeAssistant) -> COAP:
    ...

class ShellyReceiver(HomeAssistantView):
    ...

@singleton.singleton('shelly_ws_server')
async def get_ws_context(hass: HomeAssistant) -> WsServer:
    ...

def get_block_device_sleep_period(settings: dict) -> int:
    ...

def get_rpc_device_wakeup_period(status: dict) -> int:
    ...

def get_info_auth(info: dict) -> bool:
    ...

def get_info_gen(info: dict) -> int:
    ...

def get_model_name(info: dict) -> str:
    ...

def get_rpc_channel_name(device: RpcDevice, key: str) -> str:
    ...

def get_rpc_entity_name(device: RpcDevice, key: str, description: str = None) -> str:
    ...

def get_device_entry_gen(entry: ConfigEntry) -> int:
    ...

def get_rpc_key_instances(keys_dict: dict, key: str) -> List[str]:
    ...

def get_rpc_key_ids(keys_dict: dict, key: str) -> List[int]:
    ...

def is_rpc_momentary_input(config: dict, status: dict, key: str) -> bool:
    ...

def is_block_channel_type_light(settings: dict, channel: int) -> bool:
    ...

def is_rpc_channel_type_light(config: dict, channel: int) -> bool:
    ...

def is_rpc_thermostat_internal_actuator(status: dict) -> bool:
    ...

def get_rpc_input_triggers(device: RpcDevice) -> List[Tuple[str, str]]:
    ...

@callback
def update_device_fw_info(hass: HomeAssistant, shellydevice: BlockDevice, entry: ConfigEntry) -> None:
    ...

def brightness_to_percentage(brightness: int) -> int:
    ...

def percentage_to_brightness(percentage: int) -> int:
    ...

def mac_address_from_name(name: str) -> Union[str, None]:
    ...

def get_release_url(gen: int, model: str, beta: bool) -> Union[str, None]:
    ...

@callback
def async_create_issue_unsupported_firmware(hass: HomeAssistant, entry: ConfigEntry) -> None:
    ...

def is_rpc_wifi_stations_disabled(config: dict, _status: dict, key: str) -> bool:
    ...

def get_http_port(data: dict) -> int:
    ...

def get_host(host: str) -> str:
    ...

@callback
def async_remove_shelly_rpc_entities(hass: HomeAssistant, domain: str, mac: str, keys: List[str]) -> None:
    ...

def is_rpc_thermostat_mode(ident: str, status: dict) -> bool:
    ...

def get_virtual_component_ids(config: dict, platform: str) -> List[str]:
    ...

@callback
def async_remove_orphaned_entities(hass: HomeAssistant, config_entry_id: str, mac: str, platform: str, keys: List[str], key_suffix: str = None) -> None:
    ...

def get_rpc_ws_url(hass: HomeAssistant) -> Union[str, None]:
    ...

async def get_rpc_script_event_types(device: RpcDevice, id: str) -> List[str]:
    ...
