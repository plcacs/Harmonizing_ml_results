"""Tesla Fleet Data Coordinator."""
from __future__ import annotations

from datetime import datetime, timedelta
from random import randint
from time import time
from typing import TYPE_CHECKING, Any, Optional

from tesla_fleet_api import EnergySpecific, VehicleSpecific
from tesla_fleet_api.const import TeslaEnergyPeriod, VehicleDataEndpoint
from tesla_fleet_api.exceptions import (
    InvalidToken,
    LoginRequired,
    OAuthExpired,
    RateLimited,
    TeslaFleetError,
    VehicleOffline,
)
from tesla_fleet_api.ratecalculator import RateCalculator

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

if TYPE_CHECKING:
    from . import TeslaFleetConfigEntry

from .const import ENERGY_HISTORY_FIELDS, LOGGER, TeslaFleetState

VEHICLE_INTERVAL_SECONDS: int = 300
VEHICLE_INTERVAL: timedelta = timedelta(seconds=VEHICLE_INTERVAL_SECONDS)
VEHICLE_WAIT: timedelta = timedelta(minutes=15)
ENERGY_INTERVAL_SECONDS: int = 60
ENERGY_INTERVAL: timedelta = timedelta(seconds=ENERGY_INTERVAL_SECONDS)
ENERGY_HISTORY_INTERVAL: timedelta = timedelta(minutes=5)
ENDPOINTS: list[VehicleDataEndpoint] = [
    VehicleDataEndpoint.CHARGE_STATE,
    VehicleDataEndpoint.CLIMATE_STATE,
    VehicleDataEndpoint.DRIVE_STATE,
    VehicleDataEndpoint.LOCATION_DATA,
    VehicleDataEndpoint.VEHICLE_STATE,
    VehicleDataEndpoint.VEHICLE_CONFIG,
]


def flatten(data: dict[str, Any], parent: Optional[str] = None) -> dict[str, Any]:
    """Flatten the data structure."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if parent:
            key = f"{parent}_{key}"
        if isinstance(value, dict):
            result.update(flatten(value, key))
        else:
            result[key] = value
    return result


class TeslaFleetVehicleDataCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Class to manage fetching data from the TeslaFleet API."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: TeslaFleetConfigEntry,
        api: VehicleSpecific,
        product: dict[str, Any],
    ) -> None:
        """Initialize TeslaFleet Vehicle Update Coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=config_entry,
            name="Tesla Fleet Vehicle",
            update_interval=VEHICLE_INTERVAL,
        )
        self.api: VehicleSpecific = api
        self.data: dict[str, Any] = flatten(product)
        self.updated_once: bool = False
        self.last_active: datetime = datetime.now()
        self.rate: RateCalculator = RateCalculator(100, 86400, VEHICLE_INTERVAL_SECONDS, 3600, 5)

    async def _async_update_data(self) -> dict[str, Any]:
        """Update vehicle data using TeslaFleet API."""
        try:
            if self.data["state"] != TeslaFleetState.ONLINE:
                response: dict[str, Any] = await self.api.vehicle()
                self.data["state"] = response["response"]["state"]
            if self.data["state"] != TeslaFleetState.ONLINE:
                return self.data
            self.rate.consume()
            response = await self.api.vehicle_data(endpoints=ENDPOINTS)
            data: dict[str, Any] = response["response"]
        except VehicleOffline:
            self.data["state"] = TeslaFleetState.ASLEEP
            return self.data
        except RateLimited as e:
            LOGGER.warning("%s rate limited, will retry in %s seconds", self.name, e.data.get("after"))
            if "after" in e.data:
                self.update_interval = timedelta(seconds=int(e.data["after"]))
            return self.data
        except (InvalidToken, OAuthExpired, LoginRequired) as e:
            raise ConfigEntryAuthFailed from e
        except TeslaFleetError as e:
            raise UpdateFailed(e.message) from e
        self.update_interval = timedelta(seconds=self.rate.calculate())
        self.updated_once = True
        if self.api.pre2021 and data["state"] == TeslaFleetState.ONLINE:
            if (
                data["charge_state"].get("charging_state") == "Charging"
                or data["vehicle_state"].get("is_user_present")
                or data["vehicle_state"].get("sentry_mode")
            ):
                self.last_active = datetime.now()
            else:
                elapsed: timedelta = datetime.now() - self.last_active
                if elapsed > timedelta(minutes=20):
                    self.last_active = datetime.now()
                elif elapsed > timedelta(minutes=15):
                    self.update_interval = VEHICLE_WAIT
        return flatten(data)


class TeslaFleetEnergySiteLiveCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Class to manage fetching energy site live status from the TeslaFleet API."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: TeslaFleetConfigEntry,
        api: EnergySpecific,
    ) -> None:
        """Initialize TeslaFleet Energy Site Live coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=config_entry,
            name="Tesla Fleet Energy Site Live",
            update_interval=timedelta(seconds=10),
        )
        self.api: EnergySpecific = api
        self.data: dict[str, Any] = {}
        self.updated_once: bool = False

    async def _async_update_data(self) -> dict[str, Any]:
        """Update energy site data using TeslaFleet API."""
        self.update_interval = ENERGY_INTERVAL
        try:
            response: dict[str, Any] = await self.api.live_status()
            data: dict[str, Any] = response["response"]
        except RateLimited as e:
            LOGGER.warning("%s rate limited, will retry in %s seconds", self.name, e.data.get("after"))
            if "after" in e.data:
                self.update_interval = timedelta(seconds=int(e.data["after"]))
            return self.data
        except (InvalidToken, OAuthExpired, LoginRequired) as e:
            raise ConfigEntryAuthFailed from e
        except TeslaFleetError as e:
            raise UpdateFailed(e.message) from e
        data["wall_connectors"] = {wc["din"]: wc for wc in data.get("wall_connectors") or []}
        self.updated_once = True
        return data


class TeslaFleetEnergySiteHistoryCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Class to manage fetching energy site history import and export from the Tesla Fleet API."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: TeslaFleetConfigEntry,
        api: EnergySpecific,
    ) -> None:
        """Initialize Tesla Fleet Energy Site History coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=config_entry,
            name=f"Tesla Fleet Energy History {api.energy_site_id}",
            update_interval=timedelta(seconds=300),
        )
        self.api: EnergySpecific = api
        self.data: dict[str, Any] = {}
        self.updated_once: bool = False

    async def async_config_entry_first_refresh(self) -> None:
        """Set up the data coordinator."""
        await super().async_config_entry_first_refresh()
        delta: int = randint(310, 330) - int(time()) % 300
        self.logger.debug("Scheduling next %s refresh in %s seconds", self.name, delta)
        self.update_interval = timedelta(seconds=delta)
        self._schedule_refresh()
        self.update_interval = ENERGY_HISTORY_INTERVAL

    async def _async_update_data(self) -> dict[str, Any]:
        """Update energy site history data using Tesla Fleet API."""
        try:
            response: dict[str, Any] = await self.api.energy_history(TeslaEnergyPeriod.DAY)
            data: dict[str, Any] = response["response"]
        except RateLimited as e:
            LOGGER.warning("%s rate limited, will retry in %s seconds", self.name, e.data.get("after"))
            if "after" in e.data:
                self.update_interval = timedelta(seconds=int(e.data["after"]))
            return self.data
        except (InvalidToken, OAuthExpired, LoginRequired) as e:
            raise ConfigEntryAuthFailed from e
        except TeslaFleetError as e:
            raise UpdateFailed(e.message) from e
        self.updated_once = True
        output: dict[str, float] = {key: 0 for key in ENERGY_HISTORY_FIELDS}
        for period in data.get("time_series", []):
            for key in ENERGY_HISTORY_FIELDS:
                output[key] += period.get(key, 0)
        return output


class TeslaFleetEnergySiteInfoCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Class to manage fetching energy site info from the TeslaFleet API."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: TeslaFleetConfigEntry,
        api: EnergySpecific,
        product: dict[str, Any],
    ) -> None:
        """Initialize TeslaFleet Energy Info coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=config_entry,
            name="Tesla Fleet Energy Site Info",
            update_interval=timedelta(seconds=15),
        )
        self.api: EnergySpecific = api
        self.data: dict[str, Any] = flatten(product)
        self.updated_once: bool = False

    async def _async_update_data(self) -> dict[str, Any]:
        """Update energy site data using TeslaFleet API."""
        self.update_interval = ENERGY_INTERVAL
        try:
            response: dict[str, Any] = await self.api.site_info()
            data: dict[str, Any] = response["response"]
        except RateLimited as e:
            LOGGER.warning("%s rate limited, will retry in %s seconds", self.name, e.data.get("after"))
            if "after" in e.data:
                self.update_interval = timedelta(seconds=int(e.data["after"]))
            return self.data
        except (InvalidToken, OAuthExpired, LoginRequired) as e:
            raise ConfigEntryAuthFailed from e
        except TeslaFleetError as e:
            raise UpdateFailed(e.message) from e
        self.updated_once = True
        return flatten(data)