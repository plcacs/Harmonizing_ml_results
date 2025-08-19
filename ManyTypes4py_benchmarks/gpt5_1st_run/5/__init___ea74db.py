"""Read status of growatt inverters."""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal, cast

import growattServer
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_NAME, CONF_PASSWORD, CONF_URL, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import Throttle, dt as dt_util

from ..const import (
    CONF_PLANT_ID,
    DEFAULT_PLANT_ID,
    DEFAULT_URL,
    DEPRECATED_URLS,
    DOMAIN,
    LOGIN_INVALID_AUTH_CODE,
)
from .inverter import INVERTER_SENSOR_TYPES
from .mix import MIX_SENSOR_TYPES
from .sensor_entity_description import GrowattSensorEntityDescription
from .storage import STORAGE_SENSOR_TYPES
from .tlx import TLX_SENSOR_TYPES
from .total import TOTAL_SENSOR_TYPES

_LOGGER: logging.Logger = logging.getLogger(__name__)
SCAN_INTERVAL: datetime.timedelta = datetime.timedelta(minutes=5)


def get_device_list(
    api: growattServer.GrowattApi, config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], str]:
    """Retrieve the device list for the selected plant."""
    plant_id: str = config[CONF_PLANT_ID]
    login_response: Dict[str, Any] = api.login(config[CONF_USERNAME], config[CONF_PASSWORD])
    if not login_response["success"] and login_response["msg"] == LOGIN_INVALID_AUTH_CODE:
        raise ConfigEntryError("Username, Password or URL may be incorrect!")
    user_id: str = login_response["user"]["id"]
    if plant_id == DEFAULT_PLANT_ID:
        plant_info: Dict[str, Any] = api.plant_list(user_id)
        plant_id = plant_info["data"][0]["plantId"]
    devices: List[Dict[str, Any]] = api.device_list(plant_id)
    return [devices, plant_id]


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the Growatt sensor."""
    config: Dict[str, Any] = {**config_entry.data}
    username: str = config[CONF_USERNAME]
    password: str = config[CONF_PASSWORD]
    url: str = cast(str, config.get(CONF_URL, DEFAULT_URL))
    name: str = config[CONF_NAME]
    if url in DEPRECATED_URLS:
        _LOGGER.warning(
            "URL: %s has been deprecated, migrating to the latest default: %s", url, DEFAULT_URL
        )
        url = DEFAULT_URL
        config[CONF_URL] = url
        hass.config_entries.async_update_entry(config_entry, data=config)
    api: growattServer.GrowattApi = growattServer.GrowattApi(
        add_random_user_id=True, agent_identifier=username
    )
    api.server_url = url
    devices, plant_id = await hass.async_add_executor_job(get_device_list, api, config)
    probe = GrowattData(api, username, password, plant_id, "total")
    entities: List[GrowattInverter] = [
        GrowattInverter(
            probe,
            name=f"{name} Total",
            unique_id=f"{plant_id}-{description.key}",
            description=description,
        )
        for description in TOTAL_SENSOR_TYPES
    ]
    for device in devices:
        probe = GrowattData(api, username, password, device["deviceSn"], device["deviceType"])
        sensor_descriptions: Sequence[GrowattSensorEntityDescription] = ()
        if device["deviceType"] == "inverter":
            sensor_descriptions = INVERTER_SENSOR_TYPES
        elif device["deviceType"] == "tlx":
            probe.plant_id = plant_id
            sensor_descriptions = TLX_SENSOR_TYPES
        elif device["deviceType"] == "storage":
            probe.plant_id = plant_id
            sensor_descriptions = STORAGE_SENSOR_TYPES
        elif device["deviceType"] == "mix":
            probe.plant_id = plant_id
            sensor_descriptions = MIX_SENSOR_TYPES
        else:
            _LOGGER.debug(
                "Device type %s was found but is not supported right now", device["deviceType"]
            )
        entities.extend(
            [
                GrowattInverter(
                    probe,
                    name=f"{device['deviceAilas']}",
                    unique_id=f"{device['deviceSn']}-{description.key}",
                    description=description,
                )
                for description in sensor_descriptions
            ]
        )
    async_add_entities(entities, True)


class GrowattInverter(SensorEntity):
    """Representation of a Growatt Sensor."""

    _attr_has_entity_name: bool = True

    def __init__(
        self,
        probe: GrowattData,
        name: str,
        unique_id: str,
        description: GrowattSensorEntityDescription,
    ) -> None:
        """Initialize a PVOutput sensor."""
        self.probe: GrowattData = probe
        self.entity_description: GrowattSensorEntityDescription = description
        self._attr_unique_id = unique_id
        self._attr_icon = "mdi:solar-power"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, probe.device_id)},
            manufacturer="Growatt",
            name=name,
        )

    @property
    def native_value(self) -> Any:
        """Return the state of the sensor."""
        result: Any = self.probe.get_data(self.entity_description)
        if self.entity_description.precision is not None and isinstance(result, (int, float)):
            result = round(result, self.entity_description.precision)
        return result

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement of the sensor, if any."""
        if self.entity_description.currency:
            return self.probe.get_currency()
        return super().native_unit_of_measurement

    def update(self) -> None:
        """Get the latest data from the Growat API and updates the state."""
        self.probe.update()


class GrowattData:
    """The class for handling data retrieval."""

    def __init__(
        self,
        api: growattServer.GrowattApi,
        username: str,
        password: str,
        device_id: str,
        growatt_type: Literal["total", "inverter", "tlx", "storage", "mix"],
    ) -> None:
        """Initialize the probe."""
        self.growatt_type: Literal["total", "inverter", "tlx", "storage", "mix"] = growatt_type
        self.api: growattServer.GrowattApi = api
        self.device_id: str = device_id
        self.plant_id: Optional[str] = None
        self.data: Dict[str, Any] = {}
        self.previous_values: Dict[str, Any] = {}
        self.username: str = username
        self.password: str = password

    @Throttle(SCAN_INTERVAL)
    def update(self) -> None:
        """Update probe data."""
        self.api.login(self.username, self.password)
        _LOGGER.debug("Updating data for %s (%s)", self.device_id, self.growatt_type)
        try:
            if self.growatt_type == "total":
                total_info: Dict[str, Any] = self.api.plant_info(self.device_id)
                del total_info["deviceList"]
                plant_money_text, currency = total_info["plantMoneyText"].split("/")
                total_info["plantMoneyText"] = plant_money_text
                total_info["currency"] = currency
                self.data = total_info
            elif self.growatt_type == "inverter":
                inverter_info: Dict[str, Any] = self.api.inverter_detail(self.device_id)
                self.data = inverter_info
            elif self.growatt_type == "tlx":
                tlx_info: Dict[str, Any] = self.api.tlx_detail(self.device_id)
                self.data = tlx_info["data"]
            elif self.growatt_type == "storage":
                storage_info_detail: Dict[str, Any] = self.api.storage_params(self.device_id)[
                    "storageDetailBean"
                ]
                storage_energy_overview: Dict[str, Any] = self.api.storage_energy_overview(
                    self.plant_id, self.device_id
                )
                self.data = {**storage_info_detail, **storage_energy_overview}
            elif self.growatt_type == "mix":
                mix_info: Dict[str, Any] = self.api.mix_info(self.device_id)
                mix_totals: Dict[str, Any] = self.api.mix_totals(self.device_id, self.plant_id)
                mix_system_status: Dict[str, Any] = self.api.mix_system_status(
                    self.device_id, self.plant_id
                )
                mix_detail: Dict[str, Any] = self.api.mix_detail(self.device_id, self.plant_id)
                mix_chart_entries: Dict[Any, Any] = mix_detail["chartData"]
                sorted_keys = sorted(mix_chart_entries)
                date_now: datetime.date = dt_util.now().date()
                last_updated_time = dt_util.parse_time(str(sorted_keys[-1]))
                if last_updated_time is not None:
                    mix_detail["lastdataupdate"] = datetime.datetime.combine(
                        date_now, last_updated_time, dt_util.get_default_time_zone()
                    )
                dashboard_data: Dict[str, Any] = self.api.dashboard_data(self.plant_id)
                dashboard_values_for_mix: Dict[str, Any] = {
                    "etouser_combined": float(dashboard_data["etouser"].replace("kWh", ""))
                }
                self.data = {
                    **mix_info,
                    **mix_totals,
                    **mix_system_status,
                    **mix_detail,
                    **dashboard_values_for_mix,
                }
            _LOGGER.debug("Finished updating data for %s (%s)", self.device_id, self.growatt_type)
        except json.decoder.JSONDecodeError:
            _LOGGER.error("Unable to fetch data from Growatt server")

    def get_currency(self) -> Optional[str]:
        """Get the currency."""
        return self.data.get("currency")

    def get_data(self, entity_description: GrowattSensorEntityDescription) -> Any:
        """Get the data."""
        _LOGGER.debug("Data request for: %s", entity_description.name)
        variable: str = entity_description.api_key
        api_value: Any = self.data.get(variable)
        previous_value: Any = self.previous_values.get(variable)
        return_value: Any = api_value
        if (
            entity_description.previous_value_drop_threshold is not None
            and previous_value is not None
            and (api_value is not None)
        ):
            _LOGGER.debug(
                "%s - Drop threshold specified (%s), checking for drop... API Value: %s, Previous Value: %s",
                entity_description.name,
                entity_description.previous_value_drop_threshold,
                api_value,
                previous_value,
            )
            try:
                diff = float(api_value) - float(previous_value)
            except (TypeError, ValueError):
                diff = 0.0
            if -entity_description.previous_value_drop_threshold <= diff < 0:
                _LOGGER.debug(
                    "Diff is negative, but only by a small amount therefore not a nightly reset, using previous value (%s) instead of api value (%s)",
                    previous_value,
                    api_value,
                )
                return_value = previous_value
            else:
                _LOGGER.debug("%s - No drop detected, using API value", entity_description.name)
        if entity_description.never_resets and api_value == 0 and previous_value:
            _LOGGER.debug(
                "API value is 0, but this value should never reset, returning previous value (%s) instead",
                previous_value,
            )
            return_value = previous_value
        self.previous_values[variable] = return_value
        return return_value