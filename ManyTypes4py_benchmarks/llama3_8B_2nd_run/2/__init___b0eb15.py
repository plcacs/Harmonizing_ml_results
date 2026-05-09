from __future__ import annotations
import asyncio
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
import ipaddress
import logging
import os
import socket
from typing import Any, cast, ConfigType, Dict, List, Optional, Tuple
from aiohttp import web
from pyhap import util as pyhap_util
from pyhap.characteristic import Characteristic
from pyhap.const import STANDALONE_AID
from pyhap.loader import get_loader
from pyhap.service import Service
import voluptuous as vol
from zeroconf.asyncio import AsyncZeroconf

class HomeKit:
    """Class to handle all actions between HomeKit and Home Assistant."""

    def __init__(self, hass: HomeAssistant, name: str, port: int, ip_address: str, entity_filter: Dict[str, Any], exclude_accessory_mode: bool, entity_config: ConfigType, homekit_mode: str, advertise_ips: List[str], entry_id: str, entry_title: str, devices: Optional[List[str]] = None) -> None:
        ...

    async def async_start(self) -> None:
        ...

    async def async_stop(self) -> None:
        ...

    def _async_create_single_accessory(self, entity_states: List[State]) -> Optional[HomeAccessory]:
        ...

    async def _async_create_bridge_accessory(self, entity_states: List[State]) -> Optional[HomeBridge]:
        ...

    async def _async_create_accessories(self) -> bool:
        ...

    async def _async_reload_accessories(self, entity_ids: List[str]) -> None:
        ...

    async def _async_reset_accessories(self, entity_ids: List[str]) -> None:
        ...

    async def _async_recreate_removed_accessories(self, removed: List[str]) -> None:
        ...

    async def _async_update_accessories_hash(self) -> bool:
        ...

    def _async_register_bridge(self) -> None:
        ...

    async def _async_add_trigger_accessories(self) -> None:
        ...

    async def _async_create_trigger_accessory(self, device: str, device_triggers: List[Tuple[str, str]]) -> None:
        ...

    async def _async_set_device_info_attributes(self, ent_reg_ent: er.EntityRegistryEntry, dev_reg: dr.DeviceRegistry, entity_id: str) -> None:
        ...

    def _fill_config_from_device_registry_entry(self, device_entry: dr.DeviceRegistryEntry, config: Dict[str, Any]) -> None:
        ...

class HomeKitPairingQRView(HomeAssistantView):
    """Display the homekit pairing code at a protected url."""

    url = '/api/homekit/pairingqr'
    name = 'api:homekit:pairingqr'
    requires_auth = False

    async def get(self, request: web.Request) -> web.Response:
        ...
