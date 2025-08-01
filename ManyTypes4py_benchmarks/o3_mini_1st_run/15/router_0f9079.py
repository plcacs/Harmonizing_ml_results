"""Represent the Freebox router and its devices and sensors."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from freebox_api import Freepybox
from freebox_api.api.call import Call
from freebox_api.api.home import Home
from freebox_api.api.wifi import Wifi
from freebox_api.exceptions import HttpRequestError, NotOpenError
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.storage import Store
from homeassistant.util import slugify
from .const import API_VERSION, APP_DESC, CONNECTION_SENSORS_KEYS, DOMAIN, HOME_COMPATIBLE_CATEGORIES, STORAGE_KEY, STORAGE_VERSION

_LOGGER = logging.getLogger(__name__)


def is_json(json_str: str) -> bool:
    """Validate if a String is a JSON value or not."""
    try:
        json.loads(json_str)
    except (ValueError, TypeError) as err:
        _LOGGER.error("Failed to parse JSON '%s', error '%s'", json_str, err)
        return False
    return True


async def get_api(hass: HomeAssistant, host: str) -> Freepybox:
    """Get the Freebox API."""
    freebox_path: str = Store(hass, STORAGE_VERSION, STORAGE_KEY).path
    if not os.path.exists(freebox_path):
        await hass.async_add_executor_job(os.makedirs, freebox_path)
    token_file: Path = Path(f'{freebox_path}/{slugify(host)}.conf')
    return Freepybox(APP_DESC, token_file, API_VERSION)


async def get_hosts_list_if_supported(fbx_api: Freepybox) -> Tuple[bool, List[Mapping[str, Any]]]:
    """Hosts list is not supported when freebox is configured in bridge mode."""
    supports_hosts: bool = True
    fbx_devices: List[Mapping[str, Any]] = []
    try:
        fbx_devices = await fbx_api.lan.get_hosts_list() or []
    except HttpRequestError as err:
        if (matcher := re.search(r'Request failed \(APIResponse: (.+)\)', str(err))) and is_json((json_str := matcher.group(1))):
            json_resp: Mapping[str, Any] = json.loads(json_str)
            if json_resp.get('error_code') == 'nodev':
                supports_hosts = False
                _LOGGER.debug('Host list is not available using bridge mode (%s)', json_resp.get('msg'))
            else:
                raise
        else:
            raise
    return supports_hosts, fbx_devices


class FreeboxRouter:
    """Representation of a Freebox router."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, api: Freepybox, freebox_config: Mapping[str, Any]) -> None:
        """Initialize a Freebox router."""
        self.hass: HomeAssistant = hass
        self._host: str = entry.data[CONF_HOST]
        self._port: Any = entry.data[CONF_PORT]
        self._api: Freepybox = api
        self.name: str = freebox_config['model_info']['pretty_name']
        self.mac: str = freebox_config['mac']
        self._sw_v: str = freebox_config['firmware_version']
        self._attrs: Dict[str, Any] = {}
        self.supports_hosts: bool = True
        self.devices: Dict[str, Mapping[str, Any]] = {}
        self.disks: Dict[str, Mapping[str, Any]] = {}
        self.supports_raid: bool = True
        self.raids: Dict[str, Mapping[str, Any]] = {}
        self.sensors_temperature: Dict[str, Any] = {}
        self.sensors_connection: Dict[str, Any] = {}
        self.call_list: List[Any] = []
        self.home_granted: bool = True
        self.home_devices: Dict[str, Mapping[str, Any]] = {}
        self.listeners: List[Any] = []

    async def update_all(self, now: Optional[datetime] = None) -> None:
        """Update all Freebox platforms."""
        await self.update_device_trackers()
        await self.update_sensors()
        await self.update_home_devices()

    async def update_device_trackers(self) -> None:
        """Update Freebox devices."""
        new_device: bool = False
        fbx_devices: List[Mapping[str, Any]] = []
        if self.supports_hosts:
            self.supports_hosts, fbx_devices = await get_hosts_list_if_supported(self._api)
        fbx_devices.append({
            'primary_name': self.name,
            'l2ident': {'id': self.mac},
            'vendor_name': 'Freebox SAS',
            'host_type': 'router',
            'active': True,
            'attrs': self._attrs
        })
        for fbx_device in fbx_devices:
            device_mac: str = fbx_device['l2ident']['id']
            if self.devices.get(device_mac) is None:
                new_device = True
            self.devices[device_mac] = fbx_device
        async_dispatcher_send(self.hass, self.signal_device_update)
        if new_device:
            async_dispatcher_send(self.hass, self.signal_device_new)

    async def update_sensors(self) -> None:
        """Update Freebox sensors."""
        syst_datas: Mapping[str, Any] = await self._api.system.get_config()
        for sensor in syst_datas['sensors']:
            self.sensors_temperature[sensor['name']] = sensor.get('value')
        connection_datas: Mapping[str, Any] = await self._api.connection.get_status()
        for sensor_key in CONNECTION_SENSORS_KEYS:
            self.sensors_connection[sensor_key] = connection_datas[sensor_key]
        self._attrs = {
            'IPv4': connection_datas.get('ipv4'),
            'IPv6': connection_datas.get('ipv6'),
            'connection_type': connection_datas['media'],
            'uptime': datetime.fromtimestamp(round(datetime.now().timestamp()) - syst_datas['uptime_val']),
            'firmware_version': self._sw_v,
            'serial': syst_datas['serial']
        }
        self.call_list = await self._api.call.get_calls_log()
        await self._update_disks_sensors()
        await self._update_raids_sensors()
        async_dispatcher_send(self.hass, self.signal_sensor_update)

    async def _update_disks_sensors(self) -> None:
        """Update Freebox disks."""
        fbx_disks: List[Mapping[str, Any]] = await self._api.storage.get_disks() or []
        for fbx_disk in fbx_disks:
            disk: Dict[str, Any] = {**fbx_disk}
            disk_part: Dict[str, Any] = {}
            for fbx_disk_part in fbx_disk['partitions']:
                disk_part[fbx_disk_part['id']] = fbx_disk_part
            disk['partitions'] = disk_part
            self.disks[fbx_disk['id']] = disk

    async def _update_raids_sensors(self) -> None:
        """Update Freebox raids."""
        if not self.supports_raid:
            return
        try:
            fbx_raids: List[Mapping[str, Any]] = await self._api.storage.get_raids() or []
        except HttpRequestError:
            self.supports_raid = False
            _LOGGER.warning('Router %s API does not support RAID', self.name)
            return
        for fbx_raid in fbx_raids:
            self.raids[fbx_raid['id']] = fbx_raid

    async def update_home_devices(self) -> None:
        """Update Home devices (alarm, light, sensor, switch, remote ...)."""
        if not self.home_granted:
            return
        try:
            home_nodes: List[Mapping[str, Any]] = await self.home.get_home_nodes() or []
        except HttpRequestError:
            self.home_granted = False
            _LOGGER.warning('Home access is not granted')
            return
        new_device: bool = False
        for home_node in home_nodes:
            if home_node['category'] in HOME_COMPATIBLE_CATEGORIES:
                if self.home_devices.get(home_node['id']) is None:
                    new_device = True
                self.home_devices[home_node['id']] = home_node
        async_dispatcher_send(self.hass, self.signal_home_device_update)
        if new_device:
            async_dispatcher_send(self.hass, self.signal_home_device_new)

    async def reboot(self) -> None:
        """Reboot the Freebox."""
        await self._api.system.reboot()

    async def close(self) -> None:
        """Close the connection."""
        with suppress(NotOpenError):
            await self._api.close()

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device information."""
        return DeviceInfo(
            configuration_url=f'https://{self._host}:{self._port}/',
            connections={(CONNECTION_NETWORK_MAC, self.mac)},
            identifiers={(DOMAIN, self.mac)},
            manufacturer='Freebox SAS',
            name=self.name,
            sw_version=self._sw_v
        )

    @property
    def signal_device_new(self) -> str:
        """Event specific per Freebox entry to signal new device."""
        return f'{DOMAIN}-{self._host}-device-new'

    @property
    def signal_home_device_new(self) -> str:
        """Event specific per Freebox entry to signal new home device."""
        return f'{DOMAIN}-{self._host}-home-device-new'

    @property
    def signal_device_update(self) -> str:
        """Event specific per Freebox entry to signal updates in devices."""
        return f'{DOMAIN}-{self._host}-device-update'

    @property
    def signal_sensor_update(self) -> str:
        """Event specific per Freebox entry to signal updates in sensors."""
        return f'{DOMAIN}-{self._host}-sensor-update'

    @property
    def signal_home_device_update(self) -> str:
        """Event specific per Freebox entry to signal update in home devices."""
        return f'{DOMAIN}-{self._host}-home-device-update'

    @property
    def sensors(self) -> Dict[str, Any]:
        """Return sensors."""
        merged: Dict[str, Any] = {}
        merged.update(self.sensors_temperature)
        merged.update(self.sensors_connection)
        return merged

    @property
    def call(self) -> Call:
        """Return the call."""
        return self._api.call

    @property
    def wifi(self) -> Wifi:
        """Return the wifi."""
        return self._api.wifi

    @property
    def home(self) -> Home:
        """Return the home."""
        return self._api.home
