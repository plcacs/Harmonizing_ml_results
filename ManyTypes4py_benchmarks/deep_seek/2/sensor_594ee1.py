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
from typing import Any, cast, Optional, Union, List, Dict, Deque, Tuple, TypeVar

import voluptuous as vol
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.recorder import get_instance, history
from homeassistant.components.sensor import (
    DEVICE_CLASS_STATE_CLASSES, 
    DEVICE_CLASS_UNITS, 
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, 
    SensorDeviceClass, 
    SensorEntity, 
    SensorStateClass
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
    STATE_UNKNOWN
)
from homeassistant.core import (
    CALLBACK_TYPE, 
    Event, 
    EventStateChangedData, 
    EventStateReportedData, 
    HomeAssistant, 
    State, 
    callback, 
    split_entity_id
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback, 
    AddEntitiesCallback
)
from homeassistant.helpers.event import (
    async_track_point_in_utc_time, 
    async_track_state_change_event, 
    async_track_state_report_event
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util
from homeassistant.util.enum import try_parse_enum
from . import DOMAIN, PLATFORMS

_LOGGER = logging.getLogger(__name__)

STAT_AGE_COVERAGE_RATIO = 'age_coverage_ratio'
STAT_BUFFER_USAGE_RATIO = 'buffer_usage_ratio'
STAT_SOURCE_VALUE_VALID = 'source_value_valid'
STAT_AVERAGE_LINEAR = 'average_linear'
STAT_AVERAGE_STEP = 'average_step'
STAT_AVERAGE_TIMELESS = 'average_timeless'
STAT_CHANGE = 'change'
STAT_CHANGE_SAMPLE = 'change_sample'
STAT_CHANGE_SECOND = 'change_second'
STAT_COUNT = 'count'
STAT_COUNT_BINARY_ON = 'count_on'
STAT_COUNT_BINARY_OFF = 'count_off'
STAT_DATETIME_NEWEST = 'datetime_newest'
STAT_DATETIME_OLDEST = 'datetime_oldest'
STAT_DATETIME_VALUE_MAX = 'datetime_value_max'
STAT_DATETIME_VALUE_MIN = 'datetime_value_min'
STAT_DISTANCE_95P = 'distance_95_percent_of_values'
STAT_DISTANCE_99P = 'distance_99_percent_of_values'
STAT_DISTANCE_ABSOLUTE = 'distance_absolute'
STAT_MEAN = 'mean'
STAT_MEAN_CIRCULAR = 'mean_circular'
STAT_MEDIAN = 'median'
STAT_NOISINESS = 'noisiness'
STAT_PERCENTILE = 'percentile'
STAT_STANDARD_DEVIATION = 'standard_deviation'
STAT_SUM = 'sum'
STAT_SUM_DIFFERENCES = 'sum_differences'
STAT_SUM_DIFFERENCES_NONNEGATIVE = 'sum_differences_nonnegative'
STAT_TOTAL = 'total'
STAT_VALUE_MAX = 'value_max'
STAT_VALUE_MIN = 'value_min'
STAT_VARIANCE = 'variance'

T = TypeVar('T', bool, float)

def _callable_characteristic_fn(characteristic: str, binary: bool) -> Callable[[Deque[T], Deque[float], int], Optional[Union[datetime, int, float]]]:
    """Return the function callable of one characteristic function."""
    if binary:
        return STATS_BINARY_SUPPORT[characteristic]
    return STATS_NUMERIC_SUPPORT[characteristic]

def _stat_average_linear(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return states[0]
    if len(states) >= 2:
        area = 0.0
        for i in range(1, len(states)):
            area += 0.5 * (states[i] + states[i - 1]) * (ages[i] - ages[i - 1])
        age_range_seconds = ages[-1] - ages[0]
        return area / age_range_seconds
    return None

def _stat_average_step(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return states[0]
    if len(states) >= 2:
        area = 0.0
        for i in range(1, len(states)):
            area += states[i - 1] * (ages[i] - ages[i - 1])
        age_range_seconds = ages[-1] - ages[0]
        return area / age_range_seconds
    return None

def _stat_average_timeless(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    return _stat_mean(states, ages, percentile)

def _stat_change(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return states[-1] - states[0]
    return None

def _stat_change_sample(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 1:
        return (states[-1] - states[0]) / (len(states) - 1)
    return None

def _stat_change_second(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 1:
        age_range_seconds = ages[-1] - ages[0]
        if age_range_seconds > 0:
            return (states[-1] - states[0]) / age_range_seconds
    return None

def _stat_count(states: Deque[T], ages: Deque[float], percentile: int) -> int:
    return len(states)

def _stat_datetime_newest(states: Deque[T], ages: Deque[float], percentile: int) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[-1])
    return None

def _stat_datetime_oldest(states: Deque[T], ages: Deque[float], percentile: int) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[0])
    return None

def _stat_datetime_value_max(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[states.index(max(states))])
    return None

def _stat_datetime_value_min(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[states.index(min(states))])
    return None

def _stat_distance_95_percent_of_values(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) >= 1:
        return 2 * 1.96 * cast(float, _stat_standard_deviation(states, ages, percentile))
    return None

def _stat_distance_99_percent_of_values(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) >= 1:
        return 2 * 2.58 * cast(float, _stat_standard_deviation(states, ages, percentile))
    return None

def _stat_distance_absolute(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return max(states) - min(states)
    return None

def _stat_mean(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return statistics.mean(states)
    return None

def _stat_mean_circular(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        sin_sum = sum((math.sin(math.radians(x)) for x in states)
        cos_sum = sum((math.cos(math.radians(x)) for x in states)
        return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360
    return None

def _stat_median(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return statistics.median(states)
    return None

def _stat_noisiness(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return cast(float, _stat_sum_differences(states, ages, percentile)) / (len(states) - 1)
    return None

def _stat_percentile(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return states[0]
    if len(states) >= 2:
        percentiles = statistics.quantiles(states, n=100, method='exclusive')
        return percentiles[percentile - 1]
    return None

def _stat_standard_deviation(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return statistics.stdev(states)
    return None

def _stat_sum(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return sum(states)
    return None

def _stat_sum_differences(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return sum((abs(j - i) for i, j in zip(list(states), list(states)[1:], strict=False)))
    return None

def _stat_sum_differences_nonnegative(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return sum((j - i if j >= i else j - 0 for i, j in zip(list(states), list(states)[1:], strict=False)))
    return None

def _stat_total(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    return _stat_sum(states, ages, percentile)

def _stat_value_max(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return max(states)
    return None

def _stat_value_min(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return min(states)
    return None

def _stat_variance(states: Deque[float], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return 0.0
    if len(states) >= 2:
        return statistics.variance(states)
    return None

def _stat_binary_average_step(states: Deque[bool], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) == 1:
        return 100.0 * int(states[0] is True)
    if len(states) >= 2:
        on_seconds = 0.0
        for i in range(1, len(states)):
            if states[i - 1] is True:
                on_seconds += ages[i] - ages[i - 1]
        age_range_seconds = ages[-1] - ages[0]
        return 100 / age_range_seconds * on_seconds
    return None

def _stat_binary_average_timeless(states: Deque[bool], ages: Deque[float], percentile: int) -> Optional[float]:
    return _stat_binary_mean(states, ages, percentile)

def _stat_binary_count(states: Deque[bool], ages: Deque[float], percentile: int) -> int:
    return len(states)

def _stat_binary_count_on(states: Deque[bool], ages: Deque[float], percentile: int) -> int:
    return states.count(True)

def _stat_binary_count_off(states: Deque[bool], ages: Deque[float], percentile: int) -> int:
    return states.count(False)

def _stat_binary_datetime_newest(states: Deque[bool], ages: Deque[float], percentile: int) -> Optional[datetime]:
    return _stat_datetime_newest(states, ages, percentile)

def _stat_binary_datetime_oldest(states: Deque[bool], ages: Deque[float], percentile: int) -> Optional[datetime]:
    return _stat_datetime_oldest(states, ages, percentile)

def _stat_binary_mean(states: Deque[bool], ages: Deque[float], percentile: int) -> Optional[float]:
    if len(states) > 0:
        return 100.0 / len(states) * states.count(True)
    return None

STATS_NUMERIC_SUPPORT: Dict[str, Callable[[Deque[float], Deque[float], int], Optional[Union[datetime, int, float]]]] = {
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
    STAT_VARIANCE: _stat_variance
}

STATS_BINARY_SUPPORT: Dict[str, Callable[[Deque[bool], Deque[float], int], Optional[Union[datetime, int, float]]]] = {
    STAT_AVERAGE_STEP: _stat_binary_average_step,
    STAT_AVERAGE_TIMELESS: _stat_binary_average_timeless,
    STAT_COUNT: _stat_binary_count,
    STAT_COUNT_BINARY_ON: _stat_binary_count_on,
    STAT_COUNT_BINARY_OFF: _stat_binary_count_off,
    STAT_DATETIME_NEWEST: _stat_binary_datetime_newest,
    STAT_DATETIME_OLDEST: _stat_binary_datetime_oldest,
    STAT_MEAN: _stat_binary_mean
}

STATS_NOT_A_NUMBER: set[str] = {STAT_DATETIME_NEWEST, STAT_DATETIME_OLDEST, STAT_DATETIME_VALUE_MAX, STAT_DATETIME_VALUE_MIN}
STATS_DATETIME: set[str] = {STAT_DATETIME_NEWEST, STAT_DATETIME_OLDEST, STAT_DATETIME_VALUE_MAX, STAT_DATETIME_VALUE_MIN}
STATS_NUMERIC_RETAIN_UNIT: set[str] = {
    STAT_AVERAGE_LINEAR, STAT_AVERAGE_STEP, STAT_AVERAGE_TIMELESS, STAT_CHANGE, 
    STAT_DISTANCE_95P,