from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Union
from aiosolaredge import SolarEdge
from stringcase import snakecase
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from .const import DETAILS_UPDATE_DELAY, ENERGY_DETAILS_DELAY, INVENTORY_UPDATE_DELAY, LOGGER, OVERVIEW_UPDATE_DELAY, POWER_FLOW_UPDATE_DELAY
if TYPE_CHECKING:
    from .types import SolarEdgeConfigEntry

class SolarEdgeDataService(ABC):
    def __init__(self, hass: HomeAssistant, config_entry: SolarEdgeConfigEntry, api: SolarEdge, site_id: str) -> None:
        self.api: SolarEdge = api
        self.site_id: str = site_id
        self.data: Dict[str, Any] = {}
        self.attributes: Dict[str, Any] = {}
        self.hass: HomeAssistant = hass
        self.config_entry: SolarEdgeConfigEntry = config_entry

    @callback
    def async_setup(self) -> None:
        self.coordinator = DataUpdateCoordinator(self.hass, LOGGER, config_entry=self.config_entry, name=str(self), update_method=self.async_update_data, update_interval=self.update_interval)

    @property
    @abstractmethod
    def update_interval(self) -> timedelta:
        pass

    @abstractmethod
    async def async_update_data(self) -> None:
        pass

class SolarEdgeOverviewDataService(SolarEdgeDataService):
    @property
    def update_interval(self) -> timedelta:
        return OVERVIEW_UPDATE_DELAY

    async def async_update_data(self) -> None:
        pass

class SolarEdgeDetailsDataService(SolarEdgeDataService):
    @property
    def update_interval(self) -> timedelta:
        return DETAILS_UPDATE_DELAY

    async def async_update_data(self) -> None:
        pass

class SolarEdgeInventoryDataService(SolarEdgeDataService):
    @property
    def update_interval(self) -> timedelta:
        return INVENTORY_UPDATE_DELAY

    async def async_update_data(self) -> None:
        pass

class SolarEdgeEnergyDetailsService(SolarEdgeDataService):
    def __init__(self, hass: HomeAssistant, config_entry: SolarEdgeConfigEntry, api: SolarEdge, site_id: str) -> None:
        super().__init__(hass, config_entry, api, site_id)
        self.unit: Union[str, None] = None

    @property
    def update_interval(self) -> timedelta:
        return ENERGY_DETAILS_DELAY

    async def async_update_data(self) -> None:
        pass

class SolarEdgePowerFlowDataService(SolarEdgeDataService):
    def __init__(self, hass: HomeAssistant, config_entry: SolarEdgeConfigEntry, api: SolarEdge, site_id: str) -> None:
        super().__init__(hass, config_entry, api, site_id)
        self.unit: Union[str, None] = None

    @property
    def update_interval(self) -> timedelta:
        return POWER_FLOW_UPDATE_DELAY

    async def async_update_data(self) -> None:
        pass
