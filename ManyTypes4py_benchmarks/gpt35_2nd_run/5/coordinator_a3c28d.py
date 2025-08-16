from __future__ import annotations
from typing import TYPE_CHECKING, Any

class TeslaFleetVehicleDataCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    def __init__(self, hass: HomeAssistant, config_entry: TeslaFleetConfigEntry, api: Any, product: Any):
    async def _async_update_data(self) -> dict[str, Any]

class TeslaFleetEnergySiteLiveCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    def __init__(self, hass: HomeAssistant, config_entry: TeslaFleetConfigEntry, api: Any):
    async def _async_update_data(self) -> dict[str, Any]

class TeslaFleetEnergySiteHistoryCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    def __init__(self, hass: HomeAssistant, config_entry: TeslaFleetConfigEntry, api: Any):
    async def async_config_entry_first_refresh(self)
    async def _async_update_data(self) -> dict[str, Any]

class TeslaFleetEnergySiteInfoCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    def __init__(self, hass: HomeAssistant, config_entry: TeslaFleetConfigEntry, api: Any, product: Any):
    async def _async_update_data(self) -> dict[str, Any]
