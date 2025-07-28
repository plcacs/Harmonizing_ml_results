#!/usr/bin/env python3
"""Provides the data update coordinators for SolarEdge."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, TYPE_CHECKING

from aiosolaredge import SolarEdge
from stringcase import snakecase
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    DETAILS_UPDATE_DELAY,
    ENERGY_DETAILS_DELAY,
    INVENTORY_UPDATE_DELAY,
    LOGGER,
    OVERVIEW_UPDATE_DELAY,
    POWER_FLOW_UPDATE_DELAY,
)

if TYPE_CHECKING:
    from .types import SolarEdgeConfigEntry


class SolarEdgeDataService(ABC):
    """Get and update the latest data."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: SolarEdgeConfigEntry,
        api: SolarEdge,
        site_id: int,
    ) -> None:
        """Initialize the data object."""
        self.api: SolarEdge = api
        self.site_id: int = site_id
        self.data: Dict[str, Any] = {}
        self.attributes: Dict[str, Any] = {}
        self.hass: HomeAssistant = hass
        self.config_entry: SolarEdgeConfigEntry = config_entry

    @callback
    def async_setup(self) -> None:
        """Coordinator creation."""
        self.coordinator: DataUpdateCoordinator = DataUpdateCoordinator(
            self.hass,
            LOGGER,
            config_entry=self.config_entry,
            name=str(self),
            update_method=self.async_update_data,
            update_interval=self.update_interval,
        )

    @property
    @abstractmethod
    def update_interval(self) -> timedelta:
        """Update interval."""

    @abstractmethod
    async def async_update_data(self) -> None:
        """Update data."""


class SolarEdgeOverviewDataService(SolarEdgeDataService):
    """Get and update the latest overview data."""

    @property
    def update_interval(self) -> timedelta:
        """Update interval."""
        return OVERVIEW_UPDATE_DELAY

    async def async_update_data(self) -> None:
        """Update the data from the SolarEdge Monitoring API."""
        try:
            data: Dict[str, Any] = await self.api.get_overview(self.site_id)
            overview: Dict[str, Any] = data["overview"]
        except KeyError as ex:
            raise UpdateFailed("Missing overview data, skipping update") from ex

        self.data = {}
        energy_keys = ["lifeTimeData", "lastYearData", "lastMonthData", "lastDayData"]
        for key, value in overview.items():
            if key in energy_keys:
                data_val = value["energy"]
            elif key in ["currentPower"]:
                data_val = value["power"]
            else:
                data_val = value
            self.data[key] = data_val

        if set(energy_keys).issubset(self.data.keys()):
            for index, key in enumerate(energy_keys, start=1):
                if any((self.data[k] > self.data[key] for k in energy_keys[index:])):
                    LOGGER.warning("Ignoring invalid energy value %s for %s", self.data[key], key)
                    self.data.pop(key)
        LOGGER.debug("Updated SolarEdge overview: %s", self.data)


class SolarEdgeDetailsDataService(SolarEdgeDataService):
    """Get and update the latest details data."""

    @property
    def update_interval(self) -> timedelta:
        """Update interval."""
        return DETAILS_UPDATE_DELAY

    async def async_update_data(self) -> None:
        """Update the data from the SolarEdge Monitoring API."""
        try:
            data: Dict[str, Any] = await self.api.get_details(self.site_id)
            details: Dict[str, Any] = data["details"]
        except KeyError as ex:
            raise UpdateFailed("Missing details data, skipping update") from ex

        self.data = {}
        self.attributes = {}
        for key, value in details.items():
            key_snake: str = snakecase(key)
            if key_snake in ["primary_module"]:
                for module_key, module_value in value.items():
                    self.attributes[snakecase(module_key)] = module_value
            elif key_snake in ["peak_power", "type", "name", "last_update_time", "installation_date"]:
                self.attributes[key_snake] = value
            elif key_snake == "status":
                self.data["status"] = value
        LOGGER.debug("Updated SolarEdge details: %s, %s", self.data.get("status"), self.attributes)


class SolarEdgeInventoryDataService(SolarEdgeDataService):
    """Get and update the latest inventory data."""

    @property
    def update_interval(self) -> timedelta:
        """Update interval."""
        return INVENTORY_UPDATE_DELAY

    async def async_update_data(self) -> None:
        """Update the data from the SolarEdge Monitoring API."""
        try:
            data: Dict[str, Any] = await self.api.get_inventory(self.site_id)
            inventory: Dict[str, Any] = data["Inventory"]
        except KeyError as ex:
            raise UpdateFailed("Missing inventory data, skipping update") from ex

        self.data = {}
        self.attributes = {}
        for key, value in inventory.items():
            self.data[key] = len(value)
            self.attributes[key] = {key: value}
        LOGGER.debug("Updated SolarEdge inventory: %s, %s", self.data, self.attributes)


class SolarEdgeEnergyDetailsService(SolarEdgeDataService):
    """Get and update the latest power flow data."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: SolarEdgeConfigEntry,
        api: SolarEdge,
        site_id: int,
    ) -> None:
        """Initialize the power flow data service."""
        super().__init__(hass, config_entry, api, site_id)
        self.unit: Optional[str] = None

    @property
    def update_interval(self) -> timedelta:
        """Update interval."""
        return ENERGY_DETAILS_DELAY

    async def async_update_data(self) -> None:
        """Update the data from the SolarEdge Monitoring API."""
        try:
            now: datetime = datetime.now()
            today: date = date.today()
            midnight: datetime = datetime.combine(today, datetime.min.time())
            data: Dict[str, Any] = await self.api.get_energy_details(
                self.site_id, midnight, now, time_unit="DAY"
            )
            energy_details: Dict[str, Any] = data["energyDetails"]
        except KeyError as ex:
            raise UpdateFailed("Missing power flow data, skipping update") from ex

        if "meters" not in energy_details:
            LOGGER.debug("Missing meters in energy details data. Assuming site does not have any")
            return

        self.data = {}
        self.attributes = {}
        self.unit = energy_details["unit"]
        for meter in energy_details["meters"]:
            if "type" not in meter or "values" not in meter:
                continue
            if meter["type"] not in ["Production", "SelfConsumption", "FeedIn", "Purchased", "Consumption"]:
                continue
            if len(meter["values"][0]) == 2:
                self.data[meter["type"]] = meter["values"][0]["value"]
                self.attributes[meter["type"]] = {"date": meter["values"][0]["date"]}
        LOGGER.debug("Updated SolarEdge energy details: %s, %s", self.data, self.attributes)


class SolarEdgePowerFlowDataService(SolarEdgeDataService):
    """Get and update the latest power flow data."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: SolarEdgeConfigEntry,
        api: SolarEdge,
        site_id: int,
    ) -> None:
        """Initialize the power flow data service."""
        super().__init__(hass, config_entry, api, site_id)
        self.unit: Optional[str] = None

    @property
    def update_interval(self) -> timedelta:
        """Update interval."""
        return POWER_FLOW_UPDATE_DELAY

    async def async_update_data(self) -> None:
        """Update the data from the SolarEdge Monitoring API."""
        try:
            data: Dict[str, Any] = await self.api.get_current_power_flow(self.site_id)
            power_flow: Dict[str, Any] = data["siteCurrentPowerFlow"]
        except KeyError as ex:
            raise UpdateFailed("Missing power flow data, skipping update") from ex

        power_from: list[str] = []
        power_to: list[str] = []
        if "connections" not in power_flow:
            LOGGER.debug("Missing connections in power flow data. Assuming site does not have any")
            return

        for connection in power_flow["connections"]:
            power_from.append(connection["from"].lower())
            power_to.append(connection["to"].lower())

        self.data = {}
        self.attributes = {}
        self.unit = power_flow["unit"]
        for key, value in power_flow.items():
            if key in ["LOAD", "PV", "GRID", "STORAGE"]:
                self.data[key] = value.get("currentPower")
                self.attributes[key] = {"status": value["status"]}
            if key in ["GRID"]:
                export: bool = key.lower() in power_to
                if self.data[key]:
                    self.data[key] *= -1 if export else 1
                self.attributes[key]["flow"] = "export" if export else "import"
            if key in ["STORAGE"]:
                charge: bool = key.lower() in power_to
                if self.data[key]:
                    self.data[key] *= -1 if charge else 1
                self.attributes[key]["flow"] = "charge" if charge else "discharge"
                self.attributes[key]["soc"] = value["chargeLevel"]
        LOGGER.debug("Updated SolarEdge power flow: %s, %s", self.data, self.attributes)