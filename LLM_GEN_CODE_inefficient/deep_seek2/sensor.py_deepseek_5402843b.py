"""Support for deCONZ sensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar, Optional, Union, Dict, Set, List

from pydeconz.interfaces.sensors import SensorResources
from pydeconz.models.event import EventType
from pydeconz.models.sensor import SensorBase as PydeconzSensorBase
from pydeconz.models.sensor.air_purifier import AirPurifier
from pydeconz.models.sensor.air_quality import AirQuality
from pydeconz.models.sensor.carbon_dioxide import CarbonDioxide
from pydeconz.models.sensor.consumption import Consumption
from pydeconz.models.sensor.daylight import DAYLIGHT_STATUS, Daylight
from pydeconz.models.sensor.formaldehyde import Formaldehyde
from pydeconz.models.sensor.generic_status import GenericStatus
from pydeconz.models.sensor.humidity import Humidity
from pydeconz.models.sensor.light_level import LightLevel
from pydeconz.models.sensor.moisture import Moisture
from pydeconz.models.sensor.particulate_matter import ParticulateMatter
from pydeconz.models.sensor.power import Power
from pydeconz.models.sensor.pressure import Pressure
from pydeconz.models.sensor.switch import Switch
from pydeconz.models.sensor.temperature import Temperature
from pydeconz.models.sensor.time import Time

from homeassistant.components.sensor import (
    DOMAIN as SENSOR_DOMAIN,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    ATTR_TEMPERATURE,
    ATTR_VOLTAGE,
    CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    CONCENTRATION_PARTS_PER_BILLION,
    CONCENTRATION_PARTS_PER_MILLION,
    LIGHT_LUX,
    PERCENTAGE,
    EntityCategory,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfPressure,
    UnitOfTemperature,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util import dt as dt_util

from . import DeconzConfigEntry
from .const import ATTR_DARK, ATTR_ON
from .entity import DeconzDevice
from .hub import DeconzHub

PROVIDES_EXTRA_ATTRIBUTES = (
    "battery",
    "consumption",
    "daylight_status",
    "humidity",
    "light_level",
    "power",
    "pressure",
    "status",
    "temperature",
)

ATTR_CURRENT = "current"
ATTR_POWER = "power"
ATTR_DAYLIGHT = "daylight"
ATTR_EVENT_ID = "event_id"


T = TypeVar(
    "T",
    AirPurifier,
    AirQuality,
    CarbonDioxide,
    Consumption,
    Daylight,
    Formaldehyde,
    GenericStatus,
    Humidity,
    LightLevel,
    Moisture,
    ParticulateMatter,
    Power,
    Pressure,
    Temperature,
    Time,
    PydeconzSensorBase,
)


@dataclass(frozen=True, kw_only=True)
class DeconzSensorDescription(Generic[T], SensorEntityDescription):
    """Class describing deCONZ binary sensor entities."""

    instance_check: Optional[type[T]] = None
    name_suffix: str = ""
    old_unique_id_suffix: str = ""
    supported_fn: Callable[[T], bool]
    update_key: str
    value_fn: Callable[[T], Union[datetime, StateType]]


ENTITY_DESCRIPTIONS: tuple[DeconzSensorDescription, ...] = (
    DeconzSensorDescription[AirPurifier](
        key="air_purifier_filter_run_time",
        supported_fn=lambda device: True,
        update_key="filterruntime",
        name_suffix="Filter time",
        value_fn=lambda device: device.filter_run_time,
        instance_check=AirPurifier,
        device_class=SensorDeviceClass.DURATION,
        entity_category=EntityCategory.DIAGNOSTIC,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_unit_of_measurement=UnitOfTime.DAYS,
        suggested_display_precision=1,
    ),
    DeconzSensorDescription[AirQuality](
        key="air_quality",
        supported_fn=lambda device: device.supports_air_quality,
        update_key="airquality",
        value_fn=lambda device: device.air_quality,
        instance_check=AirQuality,
    ),
    DeconzSensorDescription[AirQuality](
        key="air_quality_ppb",
        supported_fn=lambda device: device.air_quality_ppb is not None,
        update_key="airqualityppb",
        value_fn=lambda device: device.air_quality_ppb,
        instance_check=AirQuality,
        name_suffix="PPB",
        old_unique_id_suffix="ppb",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_BILLION,
    ),
    DeconzSensorDescription[AirQuality](
        key="air_quality_formaldehyde",
        supported_fn=lambda device: device.air_quality_formaldehyde is not None,
        update_key="airquality_formaldehyde_density",
        value_fn=lambda device: device.air_quality_formaldehyde,
        instance_check=AirQuality,
        name_suffix="CH2O",
        device_class=SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    DeconzSensorDescription[AirQuality](
        key="air_quality_co2",
        supported_fn=lambda device: device.air_quality_co2 is not None,
        update_key="airquality_co2_density",
        value_fn=lambda device: device.air_quality_co2,
        instance_check=AirQuality,
        name_suffix="CO2",
        device_class=SensorDeviceClass.CO2,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_MILLION,
    ),
    DeconzSensorDescription[AirQuality](
        key="air_quality_pm2_5",
        supported_fn=lambda device: device.pm_2_5 is not None,
        update_key="pm2_5",
        value_fn=lambda device: device.pm_2_5,
        instance_check=AirQuality,
        name_suffix="PM25",
        device_class=SensorDeviceClass.PM25,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER,
    ),
    DeconzSensorDescription[CarbonDioxide](
        key="carbon_dioxide",
        supported_fn=lambda device: True,
        update_key="measured_value",
        value_fn=lambda device: device.carbon_dioxide,
        instance_check=CarbonDioxide,
        device_class=SensorDeviceClass.CO2,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=CONCENTRATION_PARTS_PER_BILLION,
    ),
    DeconzSensorDescription[Consumption](
        key="consumption",
        supported_fn=lambda device: device.consumption is not None,
        update_key="consumption",
        value_fn=lambda device: device.scaled_consumption,
        instance_check=Consumption,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
    ),
    Decon