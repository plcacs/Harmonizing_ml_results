"""Update coordinators for rainbird."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import datetime
import logging
from typing import Any, Optional, Set

import aiohttp
from pyrainbird.async_client import AsyncRainbirdController, RainbirdApiException, RainbirdDeviceBusyException
from pyrainbird.data import ModelAndVersion, Schedule

from homeassistant.core import HomeAssistant
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN, MANUFACTURER, TIMEOUT_SECONDS
from .types import RainbirdConfigEntry

UPDATE_INTERVAL = datetime.timedelta(minutes=1)
CALENDAR_UPDATE_INTERVAL = datetime.timedelta(minutes=15)
DEBOUNCER_COOLDOWN = 5
CONECTION_LIMIT = 1
_LOGGER = logging.getLogger(__name__)


@dataclass
class RainbirdDeviceState:
    """Data retrieved from a Rain Bird device."""
    zones: Set[int]
    active_zones: Set[int]
    rain: bool
    rain_delay: int


def async_create_clientsession() -> aiohttp.ClientSession:
    """Create a rainbird async_create_clientsession with a connection limit."""
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONECTION_LIMIT))


class RainbirdUpdateCoordinator(DataUpdateCoordinator[RainbirdDeviceState]):
    """Coordinator for rainbird API calls."""

    def __init__(self, hass: HomeAssistant, config_entry: RainbirdConfigEntry, controller: AsyncRainbirdController, model_info: ModelAndVersion) -> None:
        """Initialize RainbirdUpdateCoordinator."""
        super().__init__(hass, _LOGGER, config_entry=config_entry, name=config_entry.title, update_interval=UPDATE_INTERVAL, request_refresh_debouncer=Debouncer(hass, _LOGGER, cooldown=DEBOUNCER_COOLDOWN, immediate=False))
        self._controller: AsyncRainbirdController = controller
        self._unique_id: Optional[str] = config_entry.unique_id
        self._zones: Optional[Set[int]] = None
        self._model_info: ModelAndVersion = model_info

    @property
    def controller(self) -> AsyncRainbirdController:
        """Return the API client for the device."""
        return self._controller

    @property
    def unique_id(self) -> Optional[str]:
        """Return the config entry unique id."""
        return self._unique_id

    @property
    def device_name(self) -> str:
        """Device name for the rainbird controller."""
        return f'{MANUFACTURER} Controller'

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        """Return information about the device."""
        if self._unique_id is None:
            return None
        return DeviceInfo(name=self.device_name, identifiers={(DOMAIN, self._unique_id)}, manufacturer=MANUFACTURER, model=self._model_info.model_name, sw_version=f'{self._model_info.major}.{self._model_info.minor}')

    async def _async_update_data(self) -> RainbirdDeviceState:
        """Fetch data from Rain Bird device."""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                return await self._fetch_data()
        except RainbirdDeviceBusyException as err:
            raise UpdateFailed('Rain Bird device is busy') from err
        except RainbirdApiException as err:
            raise UpdateFailed('Rain Bird device failure') from err

    async def _fetch_data(self) -> RainbirdDeviceState:
        """Fetch data from the Rain Bird device.

        Rainbird devices can only reliably handle a single request at a time,
        so the requests are sent serially.
        """
        available_stations = await self._controller.get_available_stations()
        states = await self._controller.get_zone_states()
        rain = await self._controller.get_rain_sensor_state()
        rain_delay = await self._controller.get_rain_delay()
        return RainbirdDeviceState(zones=available_stations.active_set, active_zones=states.active_set, rain=rain, rain_delay=rain_delay)


class RainbirdScheduleUpdateCoordinator(DataUpdateCoordinator[Schedule]):
    """Coordinator for rainbird irrigation schedule calls."""

    def __init__(self, hass: HomeAssistant, config_entry: RainbirdConfigEntry, controller: AsyncRainbirdController) -> None:
        """Initialize ZoneStateUpdateCoordinator."""
        super().__init__(hass, _LOGGER, config_entry=config_entry, name=f'{config_entry.title} Schedule', update_method=self._async_update_data, update_interval=CALENDAR_UPDATE_INTERVAL)
        self._controller: AsyncRainbirdController = controller

    async def _async_update_data(self) -> Schedule:
        """Fetch data from Rain Bird device."""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                return await self._controller.get_schedule()
        except RainbirdApiException as err:
            raise UpdateFailed(f'Error communicating with Device: {err}') from err
