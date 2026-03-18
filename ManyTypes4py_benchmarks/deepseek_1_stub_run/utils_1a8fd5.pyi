```python
from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
from ipaddress import IPv4Address, IPv6Address
from types import MappingProxyType
from typing import Any, cast
from aiohttp.web import Request, WebSocketResponse
from aioshelly.block_device import COAP, Block, BlockDevice
from aioshelly.rpc_device import RpcDevice, WsServer
from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er, issue_registry as ir

def async_remove_shelly_entity(
    hass: Any,
    domain: Any,
    unique_id: Any
) -> None: ...

def get_number_of_channels(
    device: Any,
    block: Any
) -> int: ...

def get_block_entity_name(
    device: Any,
    block: Any,
    description: Any = ...
) -> str: ...

def get_block_channel_name(
    device: Any,
    block: Any
) -> str: ...

def is_block_momentary_input(
    settings: Any,
    block: Any,
    include_detached: bool = ...
) -> bool: ...

def get_device_uptime(
    uptime: Any,
    last_uptime: Any
) -> Any: ...

def get_block_input_triggers(
    device: Any,
    block: Any
) -> list[tuple[Any, Any]]: ...

def get_shbtn_input_triggers() -> list[tuple[Any, Any]]: ...

@singleton.singleton('shelly_coap')
async def get_coap_context(
    hass: Any
) -> COAP: ...

class ShellyReceiver(HomeAssistantView):
    requires_auth: bool = ...
    url: str = ...
    name: str = ...

    def __init__(self, ws_server: Any) -> None: ...
    async def get(self, request: Request) -> WebSocketResponse: ...

@singleton.singleton('shelly_ws_server')
async def get_ws_context(
    hass: Any
) -> WsServer: ...

def get_block_device_sleep_period(
    settings: Any
) -> int: ...

def get_rpc_device_wakeup_period(
    status: Any
) -> int: ...

def get_info_auth(
    info: Any
) -> bool: ...

def get_info_gen(
    info: Any
) -> int: ...

def get_model_name(
    info: Any
) -> str: ...

def get_rpc_channel_name(
    device: Any,
    key: Any
) -> str: ...

def get_rpc_entity_name(
    device: Any,
    key: Any,
    description: Any = ...
) -> str: ...

def get_device_entry_gen(
    entry: Any
) -> int: ...

def get_rpc_key_instances(
    keys_dict: Any,
    key: Any
) -> list[Any]: ...

def get_rpc_key_ids(
    keys_dict: Any,
    key: Any
) -> list[int]: ...

def is_rpc_momentary_input(
    config: Any,
    status: Any,
    key: Any
) -> bool: ...

def is_block_channel_type_light(
    settings: Any,
    channel: Any
) -> bool: ...

def is_rpc_channel_type_light(
    config: Any,
    channel: Any
) -> bool: ...

def is_rpc_thermostat_internal_actuator(
    status: Any
) -> bool: ...

def get_rpc_input_triggers(
    device: Any
) -> list[tuple[Any, Any]]: ...

@callback
def update_device_fw_info(
    hass: HomeAssistant,
    shellydevice: Any,
    entry: ConfigEntry
) -> None: ...

def brightness_to_percentage(
    brightness: Any
) -> int: ...

def percentage_to_brightness(
    percentage: Any
) -> int: ...

def mac_address_from_name(
    name: Any
) -> str | None: ...

def get_release_url(
    gen: Any,
    model: Any,
    beta: Any
) -> str | None: ...

@callback
def async_create_issue_unsupported_firmware(
    hass: HomeAssistant,
    entry: ConfigEntry
) -> None: ...

def is_rpc_wifi_stations_disabled(
    config: Any,
    _status: Any,
    key: Any
) -> bool: ...

def get_http_port(
    data: Any
) -> int: ...

def get_host(
    host: Any
) -> str: ...

@callback
def async_remove_shelly_rpc_entities(
    hass: HomeAssistant,
    domain: Any,
    mac: Any,
    keys: Any
) -> None: ...

def is_rpc_thermostat_mode(
    ident: Any,
    status: Any
) -> bool: ...

def get_virtual_component_ids(
    config: Any,
    platform: Any
) -> list[Any]: ...

@callback
def async_remove_orphaned_entities(
    hass: HomeAssistant,
    config_entry_id: Any,
    mac: Any,
    platform: Any,
    keys: Any,
    key_suffix: Any = ...
) -> None: ...

def get_rpc_ws_url(
    hass: HomeAssistant
) -> str | None: ...

async def get_rpc_script_event_types(
    device: Any,
    id: Any
) -> list[str]: ...
```