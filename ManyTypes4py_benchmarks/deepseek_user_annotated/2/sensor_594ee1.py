"""Support for statistics for sensor values."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import contextlib
from datetime import datetime, timedelta
import logging
import math
import statistics
import time
from typing import Any, cast, TypedDict, Optional, Union, Literal

import voluptuous as vol

from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.recorder import get_instance, history
from homeassistant.components.sensor import (
    DEVICE_CLASS_STATE_CLASSES,
    DEVICE_CLASS_UNITS,
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_UNIQUE_ID,
    PERCENTAGE,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    EventStateReportedData,
    HomeAssistant,
    State,
    callback,
    split_entity_id,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
    AddEntitiesCallback,
)
from homeassistant.helpers.event import (
    async_track_point_in_utc_time,
    async_track_state_change_event,
    async_track_state_report_event,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from homeassistant.util.enum import try_parse_enum

from . import DOMAIN, PLATFORMS

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Stats for attributes only
STAT_AGE_COVERAGE_RATIO: str = "age_coverage_ratio"
STAT_BUFFER_USAGE_RATIO: str = "buffer_usage_ratio"
STAT_SOURCE_VALUE_VALID: str = "source_value_valid"

# All sensor statistics
STAT_AVERAGE_LINEAR: str = "average_linear"
STAT_AVERAGE_STEP: str = "average_step"
STAT_AVERAGE_TIMELESS: str = "average_timeless"
STAT_CHANGE: str = "change"
STAT_CHANGE_SAMPLE: str = "change_sample"
STAT_CHANGE_SECOND: str = "change_second"
STAT_COUNT: str = "count"
STAT_COUNT_BINARY_ON: str = "count_on"
STAT_COUNT_BINARY_OFF: str = "count_off"
STAT_DATETIME_NEWEST: str = "datetime_newest"
STAT_DATETIME_OLDEST: str = "datetime_oldest"
STAT_DATETIME_VALUE_MAX: str = "datetime_value_max"
STAT_DATETIME_VALUE_MIN: str = "datetime_value_min"
STAT_DISTANCE_95P: str = "distance_95_percent_of_values"
STAT_DISTANCE_99P: str = "distance_99_percent_of_values"
STAT_DISTANCE_ABSOLUTE: str = "distance_absolute"
STAT_MEAN: str = "mean"
STAT_MEAN_CIRCULAR: str = "mean_circular"
STAT_MEDIAN: str = "median"
STAT_NOISINESS: str = "noisiness"
STAT_PERCENTILE: str = "percentile"
STAT_STANDARD_DEVIATION: str = "standard_deviation"
STAT_SUM: str = "sum"
STAT_SUM_DIFFERENCES: str = "sum_differences"
STAT_SUM_DIFFERENCES_NONNEGATIVE: str = "sum_differences_nonnegative"
STAT_TOTAL: str = "total"
STAT_VALUE_MAX: str = "value_max"
STAT_VALUE_MIN: str = "value_min"
STAT_VARIANCE: str = "variance"

class StatsDict(TypedDict):
    """TypedDict for statistics functions."""
    pass

def _callable_characteristic_fn(
    characteristic: str, binary: bool
) -> Callable[[deque[Union[bool, float]], deque[float], int], Optional[Union[float, int, datetime]]]:
    """Return the function callable of one characteristic function."""
    if binary:
        return STATS_BINARY_SUPPORT[characteristic]
    return STATS_NUMERIC_SUPPORT[characteristic]

def _stat_average_linear(
    states: deque[Union[bool, float]], ages: deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return states[0]
    if len(states) >= 2:
        area: float = 0
        for i in range(1, len(states)):
            area += 0.5 * (states[i] + states[i - 1]) * (ages[i] - ages[i - 1])
        age_range_seconds = ages[-1] - ages[0]
        return area / age_range_seconds
    return None

# [Rest of the statistical functions with type annotations...]

STATS_NUMERIC_SUPPORT: dict[str, Callable[[deque[Union[bool, float]], deque[float], int], Optional[Union[float, int, datetime]]]] = {
    STAT_AVERAGE_LINEAR: _stat_average_linear,
    # [Rest of the STATS_NUMERIC_SUPPORT entries...]
}

STATS_BINARY_SUPPORT: dict[str, Callable[[deque[Union[bool, float]], deque[float], int], Optional[Union[float, int, datetime]]]] = {
    STAT_AVERAGE_STEP: _stat_binary_average_step,
    # [Rest of the STATS_BINARY_SUPPORT entries...]
}

STATS_NOT_A_NUMBER: set[str] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
    STAT_DATETIME_VALUE_MIN,
}

STATS_DATETIME: set[str] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
    STAT_DATETIME_VALUE_MIN,
}

STATS_NUMERIC_RETAIN_UNIT: set[str] = {
    STAT_AVERAGE_LINEAR,
    # [Rest of the STATS_NUMERIC_RETAIN_UNIT entries...]
}

STATS_BINARY_PERCENTAGE: set[str] = {
    STAT_AVERAGE_STEP,
    STAT_AVERAGE_TIMELESS,
    STAT_MEAN,
}

CONF_STATE_CHARACTERISTIC: str = "state_characteristic"
CONF_SAMPLES_MAX_BUFFER_SIZE: str = "sampling_size"
CONF_MAX_AGE: str = "max_age"
CONF_KEEP_LAST_SAMPLE: str = "keep_last_sample"
CONF_PRECISION: str = "precision"
CONF_PERCENTILE: str = "percentile"

DEFAULT_NAME: str = "Statistical characteristic"
DEFAULT_PRECISION: int = 2
ICON: str = "mdi:calculator"

def valid_state_characteristic_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Validate that the characteristic selected is valid for the source sensor type, throw if it isn't."""
    is_binary = split_entity_id(config[CONF_ENTITY_ID])[0] == BINARY_SENSOR_DOMAIN
    characteristic = cast(str, config[CONF_STATE_CHARACTERISTIC])
    if (is_binary and characteristic not in STATS_BINARY_SUPPORT) or (
        not is_binary and characteristic not in STATS_NUMERIC_SUPPORT
    ):
        raise vol.ValueInvalid(
            f"The configured characteristic '{characteristic}' is not supported "
            "for the configured source sensor"
        )
    return config

def valid_boundary_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Validate that max_age, sampling_size, or both are provided."""
    if (
        config.get(CONF_SAMPLES_MAX_BUFFER_SIZE) is None
        and config.get(CONF_MAX_AGE) is None
    ):
        raise vol.RequiredFieldInvalid(
            "The sensor configuration must provide 'max_age' and/or 'sampling_size'"
        )
    return config

def valid_keep_last_sample(config: dict[str, Any]) -> dict[str, Any]:
    """Validate that if keep_last_sample is set, max_age must also be set."""
    if config.get(CONF_KEEP_LAST_SAMPLE) is True and config.get(CONF_MAX_AGE) is None:
        raise vol.RequiredFieldInvalid(
            "The sensor configuration must provide 'max_age' if 'keep_last_sample' is True"
        )
    return config

_PLATFORM_SCHEMA_BASE: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_ENTITY_ID): cv.entity_id,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        vol.Required(CONF_STATE_CHARACTERISTIC): cv.string,
        vol.Optional(CONF_SAMPLES_MAX_BUFFER_SIZE): vol.All(
            vol.Coerce(int), vol.Range(min=1)
        ),
        vol.Optional(CONF_MAX_AGE): cv.time_period,
        vol.Optional(CONF_KEEP_LAST_SAMPLE, default=False): cv.boolean,
        vol.Optional(CONF_PRECISION, default=DEFAULT_PRECISION): vol.Coerce(int),
        vol.Optional(CONF_PERCENTILE, default=50): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=99)
        ),
    }
)

PLATFORM_SCHEMA: vol.Schema = vol.All(
    _PLATFORM_SCHEMA_BASE,
    valid_state_characteristic_configuration,
    valid_boundary_configuration,
    valid_keep_last_sample,
)

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Statistics sensor."""
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    async_add_entities(
        new_entities=[
            StatisticsSensor(
                hass=hass,
                source_entity_id=config[CONF_ENTITY_ID],
                name=config[CONF_NAME],
                unique_id=config.get(CONF_UNIQUE_ID),
                state_characteristic=config[CONF_STATE_CHARACTERISTIC],
                samples_max_buffer_size=config.get(CONF_SAMPLES_MAX_BUFFER_SIZE),
                samples_max_age=config.get(CONF_MAX_AGE),
                samples_keep_last=config[CONF_KEEP_LAST_SAMPLE],
                precision=config[CONF_PRECISION],
                percentile=config[CONF_PERCENTILE],
            )
        ],
        update_before_add=True,
    )

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Statistics sensor entry."""
    sampling_size: Optional[int] = entry.options.get(CONF_SAMPLES_MAX_BUFFER_SIZE)
    if sampling_size:
        sampling_size = int(sampling_size)

    max_age: Optional[timedelta] = None
    if max_age := entry.options.get(CONF_MAX_AGE):
        max_age = timedelta(**max_age)

    async_add_entities(
        [
            StatisticsSensor(
                hass=hass,
                source_entity_id=entry.options[CONF_ENTITY_ID],
                name=entry.options[CONF_NAME],
                unique_id=entry.entry_id,
                state_characteristic=entry.options[CONF_STATE_CHARACTERISTIC],
                samples_max_buffer_size=sampling_size,
                samples_max_age=max_age,
                samples_keep_last=entry.options[CONF_KEEP_LAST_SAMPLE],
                precision=int(entry.options[CONF_PRECISION]),
                percentile=int(entry.options[CONF_PERCENTILE]),
            )
        ],
        True,
    )

class StatisticsSensor(SensorEntity):
    """Representation of a Statistics sensor."""

    _attr_should_poll: bool = False
    _attr_icon: str = ICON
    _attr_name: str
    _attr_unique_id: Optional[str]
    _attr_device_info: Optional[dict[str, Any]]
    _attr_available: bool
    _attr_extra_state_attributes: dict[str, Any]
    _attr_native_unit_of_measurement: Optional[str]
    _attr_device_class: Optional[SensorDeviceClass]
    _attr_state_class: Optional[SensorStateClass]
    _attr_native_value: Optional[Union[float, int, datetime, str]]

    def __init__(
        self,
        hass: HomeAssistant,
        source_entity_id: str,
        name: str,
        unique_id: Optional[str],
        state_characteristic: str,
        samples_max_buffer_size: Optional[int],
        samples_max_age: Optional[timedelta],
        samples_keep_last: bool,
        precision: int,
        percentile: int,
    ) -> None:
        """Initialize the Statistics sensor."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._source_entity_id = source_entity_id
        self._attr_device_info = async_device_info_to_link_from_entity(
            hass,
            source_entity_id,
        )
        self.is_binary: bool = (
            split_entity_id(self._source_entity_id)[0] == BINARY_SENSOR_DOMAIN
        )
        self._state_characteristic = state_characteristic
        self._samples_max_buffer_size = samples_max_buffer_size
        self._samples_max_age = (
            samples_max_age.total_seconds() if samples_max_age else None
        )
        self.samples_keep_last = samples_keep_last
        self._precision = precision
        self._percentile = percentile
        self._attr_available = False

        self.states: deque[Union[float, bool]] = deque(maxlen=samples_max_buffer_size)
        self.ages: deque[float] = deque(maxlen=samples_max_buffer_size)
        self._attr_extra_state_attributes = {}

        self._state_characteristic_fn = _callable_characteristic_fn(
            state_characteristic, self.is_binary
        )

        self._update_listener: Optional[CALLBACK_TYPE] = None
        self._preview_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None

    # [Rest of the StatisticsSensor class methods with type annotations...]
