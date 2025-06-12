"""Represent the Netgear router and its devices."""
from __future__ import annotations
import asyncio
from datetime import timedelta
import logging
from typing import Any
from pynetgear import Netgear
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_PORT, CONF_SSL, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.util import dt as dt_util
from .const import CONF_CONSIDER_HOME, DEFAULT_CONSIDER_HOME, DEFAULT_NAME, DOMAIN, MODE_ROUTER, MODELS_V2
from .errors import CannotLoginException
_LOGGER = logging.getLogger(__name__)

def get_api(password, host=None, username=None, port=None, ssl=False):
    """Get the Netgear API and login to it."""
    api = Netgear(password, host, username, port, ssl)
    if not api.login_try_port():
        raise CannotLoginException
    return api

class NetgearRouter:
    """Representation of a Netgear router."""

    def __init__(self, hass, entry):
        """Initialize a Netgear router."""
        assert entry.unique_id
        self.hass = hass
        self.entry = entry
        self.entry_id = entry.entry_id
        self.unique_id = entry.unique_id
        self._host = entry.data[CONF_HOST]
        self._port = entry.data[CONF_PORT]
        self._ssl = entry.data[CONF_SSL]
        self._username = entry.data.get(CONF_USERNAME)
        self._password = entry.data[CONF_PASSWORD]
        self._info = None
        self.model = ''
        self.mode = MODE_ROUTER
        self.device_name = ''
        self.firmware_version = ''
        self.hardware_version = ''
        self.serial_number = ''
        self.track_devices = True
        self.method_version = 1
        consider_home_int = entry.options.get(CONF_CONSIDER_HOME, DEFAULT_CONSIDER_HOME.total_seconds())
        self._consider_home = timedelta(seconds=consider_home_int)
        self.api = None
        self.api_lock = asyncio.Lock()
        self.devices = {}

    def _setup(self):
        """Set up a Netgear router sync portion."""
        self.api = get_api(self._password, self._host, self._username, self._port, self._ssl)
        self._info = self.api.get_info()
        if self._info is None:
            return False
        self.device_name = self._info.get('DeviceName', DEFAULT_NAME)
        self.model = self._info.get('ModelName')
        self.firmware_version = self._info.get('Firmwareversion')
        self.hardware_version = self._info.get('Hardwareversion')
        self.serial_number = self._info['SerialNumber']
        self.mode = self._info.get('DeviceMode', MODE_ROUTER)
        enabled_entries = [entry for entry in self.hass.config_entries.async_entries(DOMAIN) if entry.disabled_by is None]
        self.track_devices = self.mode == MODE_ROUTER or len(enabled_entries) == 1
        _LOGGER.debug("Netgear track_devices = '%s', device mode '%s'", self.track_devices, self.mode)
        for model in MODELS_V2:
            if self.model.startswith(model):
                self.method_version = 2
        if self.method_version == 2 and self.track_devices:
            if not self.api.get_attached_devices_2():
                _LOGGER.error("Netgear Model '%s' in MODELS_V2 list, but failed to get attached devices using V2", self.model)
                self.method_version = 1
        return True

    async def async_setup(self):
        """Set up a Netgear router."""
        async with self.api_lock:
            if not await self.hass.async_add_executor_job(self._setup):
                return False
        if self.track_devices:
            device_registry = dr.async_get(self.hass)
            devices = dr.async_entries_for_config_entry(device_registry, self.entry_id)
            for device_entry in devices:
                if device_entry.via_device_id is None:
                    continue
                device_mac = dict(device_entry.connections)[dr.CONNECTION_NETWORK_MAC]
                self.devices[device_mac] = {'mac': device_mac, 'name': device_entry.name, 'active': False, 'last_seen': dt_util.utcnow() - timedelta(days=365), 'device_model': None, 'device_type': None, 'type': None, 'link_rate': None, 'signal': None, 'ip': None, 'ssid': None, 'conn_ap_mac': None, 'allow_or_block': None}
        return True

    async def async_get_attached_devices(self):
        """Get the devices connected to the router."""
        if self.method_version == 1:
            async with self.api_lock:
                return await self.hass.async_add_executor_job(self.api.get_attached_devices)
        async with self.api_lock:
            return await self.hass.async_add_executor_job(self.api.get_attached_devices_2)

    async def async_update_device_trackers(self, now=None):
        """Update Netgear devices."""
        new_device = False
        ntg_devices = await self.async_get_attached_devices()
        now = dt_util.utcnow()
        if ntg_devices is None:
            return new_device
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug('Netgear scan result: \n%s', ntg_devices)
        for ntg_device in ntg_devices:
            if ntg_device.mac is None:
                continue
            device_mac = dr.format_mac(ntg_device.mac)
            if not self.devices.get(device_mac):
                new_device = True
            self.devices[device_mac] = ntg_device._asdict()
            self.devices[device_mac]['mac'] = device_mac
            self.devices[device_mac]['last_seen'] = now
        for device in self.devices.values():
            device['active'] = now - device['last_seen'] <= self._consider_home
            if not device['active']:
                device['link_rate'] = None
                device['signal'] = None
                device['ip'] = None
                device['ssid'] = None
                device['conn_ap_mac'] = None
        if new_device:
            _LOGGER.debug('Netgear tracker: new device found')
        return new_device

    async def async_get_traffic_meter(self):
        """Get the traffic meter data of the router."""
        async with self.api_lock:
            return await self.hass.async_add_executor_job(self.api.get_traffic_meter)

    async def async_get_speed_test(self):
        """Perform a speed test and get the results from the router."""
        async with self.api_lock:
            return await self.hass.async_add_executor_job(self.api.get_new_speed_test_result)

    async def async_get_link_status(self):
        """Check the ethernet link status of the router."""
        async with self.api_lock:
            return await self.hass.async_add_executor_job(self.api.check_ethernet_link)

    async def async_allow_block_device(self, mac, allow_block):
        """Allow or block a device connected to the router."""
        async with self.api_lock:
            await self.hass.async_add_executor_job(self.api.allow_block_device, mac, allow_block)

    async def async_get_utilization(self):
        """Get the system information about utilization of the router."""
        async with self.api_lock:
            return await self.hass.async_add_executor_job(self.api.get_system_info)

    async def async_reboot(self):
        """Reboot the router."""
        async with self.api_lock:
            await self.hass.async_add_executor_job(self.api.reboot)

    async def async_check_new_firmware(self):
        """Check for new firmware of the router."""
        async with self.api_lock:
            return await self.hass.async_add_executor_job(self.api.check_new_firmware)

    async def async_update_new_firmware(self):
        """Update the router to the latest firmware."""
        async with self.api_lock:
            await self.hass.async_add_executor_job(self.api.update_new_firmware)

    @property
    def port(self):
        """Port used by the API."""
        return self.api.port

    @property
    def ssl(self):
        """SSL used by the API."""
        return self.api.ssl