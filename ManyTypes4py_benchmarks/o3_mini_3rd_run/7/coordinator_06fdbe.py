from __future__ import annotations
import asyncio
from dataclasses import dataclass
import datetime
import logging
from typing import Any, Optional

import aiohttp
from aiohttp import ClientSession, TCPConnector
from pyrainbird.async_client import (
    AsyncRainbirdController,
    RainbirdApiException,
    RainbirdDeviceBusyException,
)
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
    zones: Any
    active_zones: Any
    rain: Any
    rain_delay: Any


def async_create_clientsession() -> ClientSession:
    return aiohttp.ClientSession(connector=TCPConnector(limit=CONECTION_LIMIT))


class RainbirdUpdateCoordinator(DataUpdateCoordinator[RainbirdDeviceState]):
    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: RainbirdConfigEntry,
        controller: AsyncRainbirdController,
        model_info: ModelAndVersion,
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=config_entry.title,
            update_interval=UPDATE_INTERVAL,
            request_refresh_debouncer=Debouncer(
                hass, _LOGGER, cooldown=DEBOUNCER_COOLDOWN, immediate=False
            ),
        )
        self._controller: AsyncRainbirdController = controller
        self._unique_id: Optional[str] = config_entry.unique_id
        self._zones: Any = None
        self._model_info: ModelAndVersion = model_info

    @property
    def controller(self) -> AsyncRainbirdController:
        return self._controller

    @property
    def unique_id(self) -> Optional[str]:
        return self._unique_id

    @property
    def device_name(self) -> str:
        return f"{MANUFACTURER} Controller"

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        if self._unique_id is None:
            return None
        return DeviceInfo(
            name=self.device_name,
            identifiers={(DOMAIN, self._unique_id)},
            manufacturer=MANUFACTURER,
            model=self._model_info.model_name,
            sw_version=f"{self._model_info.major}.{self._model_info.minor}",
        )

    async def _async_update_data(self) -> RainbirdDeviceState:
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                return await self._fetch_data()
        except RainbirdDeviceBusyException as err:
            raise UpdateFailed("Rain Bird device is busy") from err
        except RainbirdApiException as err:
            raise UpdateFailed("Rain Bird device failure") from err

    async def _fetch_data(self) -> RainbirdDeviceState:
        available_stations = await self._controller.get_available_stations()
        states = await self._controller.get_zone_states()
        rain = await self._controller.get_rain_sensor_state()
        rain_delay = await self._controller.get_rain_delay()
        return RainbirdDeviceState(
            zones=available_stations.active_set,
            active_zones=states.active_set,
            rain=rain,
            rain_delay=rain_delay,
        )


class RainbirdScheduleUpdateCoordinator(DataUpdateCoordinator[Schedule]):
    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: RainbirdConfigEntry,
        controller: AsyncRainbirdController,
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=f"{config_entry.title} Schedule",
            update_method=self._async_update_data,
            update_interval=CALENDAR_UPDATE_INTERVAL,
        )
        self._controller: AsyncRainbirdController = controller

    async def _async_update_data(self) -> Schedule:
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                return await self._controller.get_schedule()
        except RainbirdApiException as err:
            raise UpdateFailed(f"Error communicating with Device: {err}") from err