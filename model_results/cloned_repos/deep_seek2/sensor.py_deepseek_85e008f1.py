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
from typing import Any, cast, Optional, Union, List, Dict, Deque

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


def _callable_characteristic_fn(
    characteristic: str, binary: bool
) -> Callable[[Deque[Union[bool, float]], Deque[float], int], Union[float, int, datetime, None]]:
    """Return the function callable of one characteristic function."""
    if binary:
        return STATS_BINARY_SUPPORT[characteristic]
    return STATS_NUMERIC_SUPPORT[characteristic]


# Statistics for numeric sensor


def _stat_average_linear(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
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


def _stat_average_step(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return states[0]
    if len(states) >= 2:
        area: float = 0
        for i in range(1, len(states)):
            area += states[i - 1] * (ages[i] - ages[i - 1])
        age_range_seconds = ages[-1] - ages[0]
        return area / age_range_seconds
    return None


def _stat_average_timeless(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    return _stat_mean(states, ages, percentile)


def _stat_change(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return states[-1] - states[0]
    return None


def _stat_change_sample(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 1:
        return (states[-1] - states[0]) / (len(states) - 1)
    return None


def _stat_change_second(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 1:
        age_range_seconds = ages[-1] - ages[0]
        if age_range_seconds > 0:
            return (states[-1] - states[0]) / age_range_seconds
    return None


def _stat_count(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[int]:
    return len(states)


def _stat_datetime_newest(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[-1])
    return None


def _stat_datetime_oldest(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[0])
    return None


def _stat_datetime_value_max(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[states.index(max(states))])
    return None


def _stat_datetime_value_min(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[states.index(min(states))])
    return None


def _stat_distance_95_percent_of_values(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) >= 1:
        return (
            2 * 1.96 * cast(float, _stat_standard_deviation(states, ages, percentile))
        )
    return None


def _stat_distance_99_percent_of_values(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) >= 1:
        return (
            2 * 2.58 * cast(float, _stat_standard_deviation(states, ages, percentile))
        )
    return None


def _stat_distance_absolute(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return max(states) - min(states)
    return None


def _stat_mean(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return statistics.mean(states)
    return None


def _stat_mean_circular(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        sin_sum = sum(math.sin(math.radians(x)) for x in states)
        cos_sum = sum(math.cos(math.radians(x)) for x in states)
        return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360
    return None


def _stat_median(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return statistics.median(states)
    return None


def _stat_noisiness(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return cast(float, _stat_sum_differences(states, ages, percentile)) / (
            len(states) - 1
        )
    return None


def _stat_percentile(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return states[0]
    if len(states) >= 2:
        percentiles = statistics.quantiles(states, n=100, method="exclusive")
        return percentiles[percentile - 1]
    return None


def _stat_standard_deviation(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return statistics.stdev(states)
    return None


def _stat_sum(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return sum(states)
    return None


def _stat_sum_differences(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return sum(
            abs(j - i) for i, j in zip(list(states), list(states)[1:], strict=False)
        )
    return None


def _stat_sum_differences_nonnegative(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return sum(
            (j - i if j >= i else j - 0)
            for i, j in zip(list(states), list(states)[1:], strict=False)
        )
    return None


def _stat_total(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    return _stat_sum(states, ages, percentile)


def _stat_value_max(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return max(states)
    return None


def _stat_value_min(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return min(states)
    return None


def _stat_variance(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return statistics.variance(states)
    return None


# Statistics for binary sensor


def _stat_binary_average_step(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) == 1:
        return 100.0 * int(states[0] is True)
    if len(states) >= 2:
        on_seconds: float = 0
        for i in range(1, len(states)):
            if states[i - 1] is True:
                on_seconds += ages[i] - ages[i - 1]
        age_range_seconds = ages[-1] - ages[0]
        return 100 / age_range_seconds * on_seconds
    return None


def _stat_binary_average_timeless(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    return _stat_binary_mean(states, ages, percentile)


def _stat_binary_count(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[int]:
    return len(states)


def _stat_binary_count_on(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[int]:
    return states.count(True)


def _stat_binary_count_off(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[int]:
    return states.count(False)


def _stat_binary_datetime_newest(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[datetime]:
    return _stat_datetime_newest(states, ages, percentile)


def _stat_binary_datetime_oldest(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[datetime]:
    return _stat_datetime_oldest(states, ages, percentile)


def _stat_binary_mean(
    states: Deque[Union[bool, float]], ages: Deque[float], percentile: int
) -> Optional[float]:
    if len(states) > 0:
        return 100.0 / len(states) * states.count(True)
    return None


# Statistics supported by a sensor source (numeric)
STATS_NUMERIC_SUPPORT: Dict[str, Callable[[Deque[Union[bool, float]], Deque[float], int], Union[float, int, datetime, None]]] = {
    STAT_AVERAGE_LINEAR: _stat_average_linear,
    STAT_AVERAGE_STEP: _stat_average_step,
    STAT_AVERAGE_TIMELESS: _stat_average_timeless,
    STAT_CHANGE_SAMPLE: _stat_change_sample,
    STAT_CHANGE_SECOND: _stat_change_second,
    STAT_CHANGE: _stat_change,
    STAT_COUNT: _stat_count,
    STAT_DATETIME_NEWEST: _stat_datetime_newest,
    STAT_DATETIME_OLDEST: _stat_datetime_oldest,
    STAT_DATETIME_VALUE_MAX: _stat_datetime_value_max,
    STAT_DATETIME_VALUE_MIN: _stat_datetime_value_min,
    STAT_DISTANCE_95P: _stat_distance_95_percent_of_values,
    STAT_DISTANCE_99P: _stat_distance_99_percent_of_values,
    STAT_DISTANCE_ABSOLUTE: _stat_distance_absolute,
    STAT_MEAN: _stat_mean,
    STAT_MEAN_CIRCULAR: _stat_mean_circular,
    STAT_MEDIAN: _stat_median,
    STAT_NOISINESS: _stat_noisiness,
    STAT_PERCENTILE: _stat_percentile,
    STAT_STANDARD_DEVIATION: _stat_standard_deviation,
    STAT_SUM: _stat_sum,
    STAT_SUM_DIFFERENCES: _stat_sum_differences,
    STAT_SUM_DIFFERENCES_NONNEGATIVE: _stat_sum_differences_nonnegative,
    STAT_TOTAL: _stat_total,
    STAT_VALUE_MAX: _stat_value_max,
    STAT_VALUE_MIN: _stat_value_min,
    STAT_VARIANCE: _stat_variance,
}

# Statistics supported by a binary_sensor source
STATS_BINARY_SUPPORT: Dict[str, Callable[[Deque[Union[bool, float]], Deque[float], int], Union[float, int, datetime, None]]] = {
    STAT_AVERAGE_STEP: _stat_binary_average_step,
    STAT_AVERAGE_TIMELESS: _stat_binary_average_timeless,
    STAT_COUNT: _stat_binary_count,
    STAT_COUNT_BINARY_ON: _stat_binary_count_on,
    STAT_COUNT_BINARY_OFF: _stat_binary_count_off,
    STAT_DATETIME_NEWEST: _stat_binary_datetime_newest,
    STAT_DATETIME_OLDEST: _stat_binary_datetime_oldest,
    STAT_MEAN: _stat_binary_mean,
}

STATS_NOT_A_NUMBER: Set[str] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
    STAT_DATETIME_VALUE_MIN,
}

STATS_DATETIME: Set[str] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
    STAT_DATETIME_VALUE_MIN,
}

# Statistics which retain the unit of the source entity
STATS_NUMERIC_RETAIN_UNIT: Set[str] = {
    STAT_AVERAGE_LINEAR,
    STAT_AVERAGE_STEP,
    STAT_AVERAGE_TIMELESS,
    STAT_CHANGE,
    STAT_DISTANCE_95P,
    STAT_DISTANCE_99P,
    STAT_DISTANCE_ABSOLUTE,
    STAT_MEAN,
    STAT_MEAN_CIRCULAR,
    STAT_MEDIAN,
    STAT_NOISINESS,
    STAT_PERCENTILE,
    STAT_STANDARD_DEVIATION,
    STAT_SUM,
    STAT_SUM_DIFFERENCES,
    STAT_SUM_DIFFERENCES_NONNEGATIVE,
    STAT_TOTAL,
    STAT_VALUE_MAX,
    STAT_VALUE_MIN,
}

# Statistics which produce percentage ratio from binary_sensor source entity
STATS_BINARY_PERCENTAGE: Set[str] = {
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


def valid_state_characteristic_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
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


def valid_boundary_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that max_age, sampling_size, or both are provided."""

    if (
        config.get(CONF_SAMPLES_MAX_BUFFER_SIZE) is None
        and config.get(CONF_MAX_AGE) is None
    ):
        raise vol.RequiredFieldInvalid(
            "The sensor configuration must provide 'max_age' and/or 'sampling_size'"
        )
    return config


def valid_keep_last_sample(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that if keep_last_sample is set, max_age must also be set."""

    if config.get(CONF_KEEP_LAST_SAMPLE) is True and config.get(CONF_MAX_AGE) is None:
        raise vol.RequiredFieldInvalid(
            "The sensor configuration must provide 'max_age' if 'keep_last_sample' is True"
        )
    return config


_PLATFORM_SCHEMA_BASE = SENSOR_PLATFORM_SCHEMA.extend(
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
PLATFORM_SCHEMA = vol.All(
    _PLATFORM_SCHEMA_BASE,
    valid_state_characteristic_configuration,
    valid_boundary_configuration,
    valid_keep_last_sample,
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
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
    sampling_size = entry.options.get(CONF_SAMPLES_MAX_BUFFER_SIZE)
    if sampling_size:
        sampling_size = int(sampling_size)

    max_age = None
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
        self._attr_name: str = name
        self._attr_unique_id: Optional[str] = unique_id
        self._source_entity_id: str = source_entity_id
        self._attr_device_info = async_device_info_to_link_from_entity(
            hass,
            source_entity_id,
        )
        self.is_binary: bool = (
            split_entity_id(self._source_entity_id)[0] == BINARY_SENSOR_DOMAIN
        )
        self._state_characteristic: str = state_characteristic
        self._samples_max_buffer_size: Optional[int] = samples_max_buffer_size
        self._samples_max_age: Optional[float] = (
            samples_max_age.total_seconds() if samples_max_age else None
        )
        self.samples_keep_last: bool = samples_keep_last
        self._precision: int = precision
        self._percentile: int = percentile
        self._attr_available: bool = False

        self.states: Deque[Union[float, bool]] = deque(maxlen=samples_max_buffer_size)
        self.ages: Deque[float] = deque(maxlen=samples_max_buffer_size)
        self._attr_extra_state_attributes: Dict[str, Any] = {}

        self._state_characteristic_fn: Callable[
            [Deque[Union[bool, float]], Deque[float], int],
            Union[float, int, datetime, None],
        ] = _callable_characteristic_fn(state_characteristic, self.is_binary)

        self._update_listener: Optional[CALLBACK_TYPE] = None
        self._preview_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None

    async def async_start_preview(
        self,
        preview_callback: Callable[[str, Mapping[str, Any]], None],
    ) -> CALLBACK_TYPE:
        """Render a preview."""
        # abort early if there is no entity_id
        # as without we can't track changes
        # or either size or max_age is not set
        if not self._source_entity_id or (
            self._samples_max_buffer_size is None and self._samples_max_age is None
        ):
            self._attr_available = False
            calculated_state = self._async_calculate_state()
            preview_callback(calculated_state.state, calculated_state.attributes)
            return self._call_on_remove_callbacks

        self._preview_callback = preview_callback

        await self._async_stats_sensor_startup()
        return self._call_on_remove_callbacks

    def _async_handle_new_state(
        self,
        reported_state: Optional[State],
    ) -> None:
        """Handle the sensor state changes."""
        if (new_state := reported_state) is None:
            return
        self._add_state_to_queue(new_state)
        self._async_purge_update_and_schedule()

        if self._preview_callback:
            calculated_state = self._async_calculate_state()
            self._preview_callback(calculated_state.state, calculated_state.attributes)
        # only write state to the state machine if we are not in preview mode
        if not self._preview_callback:
            self.async_write_ha_state()

    @callback
    def _async_stats_sensor_state_change_listener(
        self,
        event: Event[EventStateChangedData],
    ) -> None:
        self._async_handle_new_state(event.data["new_state"])

    @callback
    def _async_stats_sensor_state_report_listener(
        self,
        event: Event[EventStateReportedData],
    ) -> None:
        self._async_handle_new_state(event.data["new_state"])

    async def _async_stats_sensor_startup(self) -> None:
        """Add listener and get recorded state.

        Historical data needs to be loaded from the database first before we
        can start accepting new incoming changes.
        This is needed to ensure that the buffer is properly sorted by time.
        """
        _LOGGER.debug("Startup for %s", self.entity_id)
        if "recorder" in self.hass.config.components:
            await self._initialize_from_database()
        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                [self._source_entity_id],
                self._async_stats_sensor_state_change_listener,
            )
        )
        self.async_on_remove(
            async_track_state_report_event(
                self.hass,
                [self._source_entity_id],
                self._async_stats_sensor_state_report_listener,
            )
        )

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        await self._async_stats_sensor_startup()

    def _add_state_to_queue(self, new_state: State) -> None:
        """Add the state to the queue."""

        # Attention: it is not safe to store the new_state object,
        # since the "last_reported" value will be updated over time.
        # Here we make a copy the current value, which is okay.
        self._attr_available = new_state.state != STATE_UNAVAILABLE
        if new_state.state == STATE_UNAVAILABLE:
            self._attr_extra_state_attributes[STAT_SOURCE_VALUE_VALID] = None
            return
        if new_state.state in (STATE_UNKNOWN, None, ""):
            self._attr_extra_state_attributes[STAT_SOURCE_VALUE_VALID] = False
            return

        try:
            if self.is_binary:
                assert new_state.state in ("on", "off")
                self.states.append(new_state.state == "on")
            else:
                self.states.append(float(new_state.state))
            self.ages.append(new_state.last_reported_timestamp)
            self._attr_extra_state_attributes[STAT_SOURCE_VALUE_VALID] = True
        except ValueError:
            self._attr_extra_state_attributes[STAT_SOURCE_VALUE_VALID] = False
            _LOGGER.error(
                "%s: parsing error. Expected number or binary state, but received '%s'",
                self.entity_id,
                new_state.state,
            )
            return

        self._calculate_state_attributes(new_state)

    def _calculate_state_attributes(self, new_state: State) -> None:
        """Set the entity state attributes."""

        self._attr_native_unit_of_measurement = self._calculate_unit_of_measurement(
            new_state
        )
        self._attr_device_class = self._calculate_device_class(
            new_state, self._attr_native_unit_of_measurement
        )
        self._attr_state_class = self._calculate_state_class(new_state)

    def _calculate_unit_of_measurement(self, new_state: State) -> Optional[str]:
        """Return the calculated unit of measurement.

        The unit of measurement is that of the source sensor, adjusted based on the
        state characteristics.
        """

        base_unit: Optional[str] = new_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        unit: Optional[str] = None
        stat_type = self._state_characteristic
        if self.is_binary and stat_type in STATS_BINARY_PERCENTAGE:
            unit = PERCENTAGE
        elif not base_unit:
            unit = None
        elif stat_type in STATS_NUMERIC_RETAIN_UNIT:
            unit = base_unit
        elif stat_type in STATS_NOT_A_NUMBER or stat_type in (
            STAT_COUNT,
            STAT_COUNT_BINARY_ON,
            STAT_COUNT_BINARY_OFF,
        ):
            unit = None
        elif stat_type == STAT_VARIANCE:
            unit = base_unit + "²"
        elif stat_type == STAT_CHANGE_SAMPLE:
            unit = base_unit + "/sample"
        elif stat_type == STAT_CHANGE_SECOND:
            unit = base_unit + "/s"

        return unit

    def _calculate_device_class(
        self, new_state: State, unit: Optional[str]
    ) -> Optional[SensorDeviceClass]:
        """Return the calculated device class.

        The device class is calculated based on the state characteristics,
        the source device class and the unit of measurement is
        in the device class units list.
        """

        device_class: Optional[SensorDeviceClass] = None
        stat_type = self._state_characteristic
        if stat_type in STATS_DATETIME:
            return SensorDeviceClass.TIMESTAMP
        if stat_type in STATS_NUMERIC_RETAIN_UNIT:
            device_class = new_state.attributes.get(ATTR_DEVICE_CLASS)
            if device_class is None:
                return None
            if (
                sensor_device_class := try_parse_enum(SensorDeviceClass, device_class)
            ) is None:
                return None
            if (
                sensor_device_class
                and (
                    sensor_state_classes := DEVICE_CLASS_STATE_CLASSES.get(
                        sensor_device_class
                    )
                )
                and sensor_state_classes
                and SensorStateClass.MEASUREMENT not in sensor_state_classes
            ):
                return None
            if device_class not in DEVICE_CLASS_UNITS:
                return None
            if (
                device_class in DEVICE_CLASS_UNITS
                and unit not in DEVICE_CLASS_UNITS[device_class]
            ):
                return None

        return device_class

    def _calculate_state_class(self, new_state: State) -> Optional[SensorState