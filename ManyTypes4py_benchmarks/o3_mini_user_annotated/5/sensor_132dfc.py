from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyecobee.const import ECOBEE_STATE_CALIBRATING, ECOBEE_STATE_UNKNOWN

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    CONCENTRATION_PARTS_PER_MILLION,
    PERCENTAGE,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import EcobeeConfigEntry
from .const import DOMAIN, ECOBEE_MODEL_TO_NAME, MANUFACTURER


@dataclass(frozen=True, kw_only=True)
class EcobeeSensorEntityDescription(SensorEntityDescription):
    runtime_key: str | None


SENSOR_TYPES: tuple[EcobeeSensorEntityDescription, ...] = (
    EcobeeSensorEntityDescription(
        key="temperature",
        native_unit_of_measurement=UnitOfTemperature.FAHRENHEIT,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        runtime_key=None,
    ),
    EcobeeSensorEntityDescription(
        key="humidity",
        native_unit_of_measurement=PERCENTAGE,
        device_class=SensorDeviceClass.HUMIDITY,
        state_class=SensorStateClass.MEASUREMENT,
        runtime_key=None,
    ),
    EcobeeSensorEntityDescription(
        key="co2PPM",
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_MILLION,
        device_class=SensorDeviceClass.CO2,
        state_class=SensorStateClass.MEASUREMENT,
        runtime_key="actualCO2",
    ),
    EcobeeSensorEntityDescription(
        key="vocPPM",
        device_class=SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
        state_class=SensorStateClass.MEASUREMENT,
        runtime_key="actualVOC",
    ),
    EcobeeSensorEntityDescription(
        key="airQuality",
        device_class=SensorDeviceClass.AQI,
        state_class=SensorStateClass.MEASUREMENT,
        runtime_key="actualAQScore",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: EcobeeConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    data: Any = config_entry.runtime_data
    entities = [
        EcobeeSensor(data, sensor["name"], index, description)
        for index in range(len(data.ecobee.thermostats))
        for sensor in data.ecobee.get_remote_sensors(index)
        for item in sensor["capability"]
        for description in SENSOR_TYPES
        if description.key == item["type"]
    ]

    async_add_entities(entities, True)


class EcobeeSensor(SensorEntity):
    _attr_has_entity_name: bool = True
    entity_description: EcobeeSensorEntityDescription

    def __init__(
        self, data: Any, sensor_name: str, sensor_index: int, description: EcobeeSensorEntityDescription
    ) -> None:
        self.entity_description = description
        self.data: Any = data
        self.sensor_name: str = sensor_name
        self.index: int = sensor_index
        self._state: Any = None

    @property
    def unique_id(self) -> str | None:
        for sensor in self.data.ecobee.get_remote_sensors(self.index):
            if sensor["name"] == self.sensor_name:
                if "code" in sensor:
                    return f"{sensor['code']}-{self.device_class}"
                thermostat: dict[str, Any] = self.data.ecobee.get_thermostat(self.index)
                return f"{thermostat['identifier']}-{sensor['id']}-{self.device_class}"
        return None

    @property
    def device_info(self) -> DeviceInfo | None:
        identifier: str | None = None
        model: str | None = None
        for sensor in self.data.ecobee.get_remote_sensors(self.index):
            if sensor["name"] != self.sensor_name:
                continue
            if "code" in sensor:
                identifier = sensor["code"]
                model = "ecobee Room Sensor"
            else:
                thermostat: dict[str, Any] = self.data.ecobee.get_thermostat(self.index)
                identifier = thermostat["identifier"]
                try:
                    model = f"{ECOBEE_MODEL_TO_NAME[thermostat['modelNumber']]} Thermostat"
                except KeyError:
                    model = None
            break

        if identifier is not None and model is not None:
            return DeviceInfo(
                identifiers={(DOMAIN, identifier)},
                manufacturer=MANUFACTURER,
                model=model,
                name=self.sensor_name,
            )
        return None

    @property
    def available(self) -> bool:
        thermostat: dict[str, Any] = self.data.ecobee.get_thermostat(self.index)
        return thermostat["runtime"]["connected"]

    @property
    def native_value(self) -> float | Any | None:
        if self._state in (ECOBEE_STATE_CALIBRATING, ECOBEE_STATE_UNKNOWN, "unknown"):
            return None

        if self.entity_description.key == "temperature":
            return float(self._state) / 10

        return self._state

    async def async_update(self) -> None:
        await self.data.update()
        for sensor in self.data.ecobee.get_remote_sensors(self.index):
            if sensor["name"] != self.sensor_name:
                continue
            for item in sensor["capability"]:
                if item["type"] != self.entity_description.key:
                    continue
                if self.entity_description.runtime_key is None:
                    self._state = item["value"]
                else:
                    thermostat: dict[str, Any] = self.data.ecobee.get_thermostat(self.index)
                    self._state = thermostat["runtime"][self.entity_description.runtime_key]
                break