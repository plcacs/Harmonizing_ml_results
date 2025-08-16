from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Set
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

@dataclass
class OndiloIcoPoolData:
    measures_coordinator: OndiloIcoMeasuresCoordinator = field(init=False)

@dataclass
class OndiloIcoMeasurementData:
    sensors: Dict[str, Any]

class OndiloIcoPoolsCoordinator(DataUpdateCoordinator[Dict[str, OndiloIcoPoolData]]):
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, api: OndiloClient):
        ...

    async def _async_update_data(self) -> Dict[str, OndiloIcoPoolData]:
        ...

    def _update_data(self) -> Dict[str, OndiloIcoPoolData]:
        ...

class OndiloIcoMeasuresCoordinator(DataUpdateCoordinator[OndiloIcoMeasurementData]):
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, api: OndiloClient, pool_id: str):
        ...

    async def _async_update_data(self) -> OndiloIcoMeasurementData:
        ...

    def _update_data(self) -> OndiloIcoMeasurementData:
        ...

    def set_next_refresh(self, pool_data: OndiloIcoPoolData):
        ...
