from __future__ import annotations
from collections.abc import Callable, Mapping
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Any
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.storage import Store
from .const import API_VERSION, APP_DESC, CONNECTION_SENSORS_KEYS, DOMAIN, HOME_COMPATIBLE_CATEGORIES, STORAGE_KEY, STORAGE_VERSION
_LOGGER: logging.Logger

def is_json(json_str: str) -> bool:
    ...

async def get_api(hass: HomeAssistant, host: str) -> Freepybox:
    ...

async def get_hosts_list_if_supported(fbx_api: Freepybox) -> tuple[bool, list]:
    ...

class FreeboxRouter:
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, api: Freepybox, freebox_config: dict[str, Any]):
        ...

    async def update_all(self, now: datetime = None) -> None:
        ...

    async def update_device_trackers(self) -> None:
        ...

    async def update_sensors(self) -> None:
        ...

    async def _update_disks_sensors(self) -> None:
        ...

    async def _update_raids_sensors(self) -> None:
        ...

    async def update_home_devices(self) -> None:
        ...

    async def reboot(self) -> None:
        ...

    async def close(self) -> None:
        ...

    @property
    def device_info(self) -> DeviceInfo:
        ...

    @property
    def signal_device_new(self) -> str:
        ...

    @property
    def signal_home_device_new(self) -> str:
        ...

    @property
    def signal_device_update(self) -> str:
        ...

    @property
    def signal_sensor_update(self) -> str:
        ...

    @property
    def signal_home_device_update(self) -> str:
        ...

    @property
    def sensors(self) -> dict[str, Any]:
        ...

    @property
    def call(self) -> Call:
        ...

    @property
    def wifi(self) -> Wifi:
        ...

    @property
    def home(self) -> Home:
        ...
