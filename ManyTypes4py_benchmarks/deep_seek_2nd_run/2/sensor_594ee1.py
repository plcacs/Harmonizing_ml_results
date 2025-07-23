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
from typing import Any, cast, Optional, Union, List, Dict, Tuple, Deque

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

_LOGGER = logging.getLogger(__name__)

STAT_AGE_COVERAGE_RATIO: str = 'age_coverage_ratio'
STAT_BUFFER_USAGE_RATIO: str = 'buffer_usage_ratio'
STAT_SOURCE_VALUE_VALID: str = 'source_value_valid'
STAT_AVERAGE_LINEAR: str = 'average_linear'
STAT_AVERAGE_STEP: str = 'average_step'
STAT_AVERAGE_TIMELESS: str = 'average_timeless'
STAT_CHANGE: str = 'change'
STAT_CHANGE_SAMPLE: str = 'change_sample'
STAT_CHANGE_SECOND: str = 'change_second'
STAT_COUNT: str = 'count'
STAT_COUNT_BINARY_ON: str = 'count_on'
STAT_COUNT_BINARY_OFF: str = 'count_off'
STAT_DATETIME_NEWEST: str = 'datetime_newest'
STAT_DATETIME_OLDEST: str = 'datetime_oldest'
STAT_DATETIME_VALUE_MAX: str = 'datetime_value_max'
STAT_DATETIME_VALUE_MIN: str = 'datetime_value_min'
STAT_DISTANCE_95P: str = 'distance_95_percent_of_values'
STAT_DISTANCE_99P: str = 'distance_99_percent_of_values'
STAT_DISTANCE_ABSOLUTE: str = 'distance_absolute'
STAT_MEAN: str = 'mean'
STAT_MEAN_CIRCULAR: str = 'mean_circular'
STAT_MEDIAN: str = 'median'
STAT_NOISINESS: str = 'noisiness'
STAT_PERCENTILE: str = 'percentile'
STAT_STANDARD_DEVIATION: str = 'standard_deviation'
STAT_SUM: str = 'sum'
STAT_SUM_DIFFERENCES: str = 'sum_differences'
STAT_SUM_DIFFERENCES_NONNEGATIVE: str = 'sum_differences_nonnegative'
STAT_TOTAL: str = 'total'
STAT_VALUE_MAX: str = 'value_max'
STAT_VALUE_MIN: str = 'value_min'
STAT_VARIANCE: str = 'variance'

def _callable_characteristic_fn(characteristic: str, binary: bool) -> Callable[[Deque[Union[bool, float]], Deque[float], int], Optional[Union[datetime, int, float]]]:
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

def _stat_count(states: Deque[Union[bool, float]], ages: Deque[float], percentile: int) -> int:
    return len(states)

def _stat_datetime_newest(states: Deque[Union[bool, float]], ages: Deque[float], percentile: int) -> Optional[datetime]:
    if len(states) > 0:
        return dt_util.utc_from_timestamp(ages[-1])
    return None

def _stat_datetime_oldest(states: Deque[Union[bool, float]], ages: Deque[float], percentile: int) -> Optional[datetime]:
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
        sin_sum = sum((math.sin(math.radians(x)) for x in states))
        cos_sum = sum((math.cos(math.radians(x)) for x in states))
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

STATS_NUMERIC_SUPPORT: Dict[str, Callable[[Deque[float], Deque[float], int], Optional[Union[datetime, float]]]] = {
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

STATS_BINARY_SUPPORT: Dict[str, Callable[[Deque[bool], Deque[float], int], Optional[Union[datetime, float, int]]]] = {
    STAT_AVERAGE_STEP: _stat_binary_average_step,
    STAT_AVERAGE_TIMELESS: _stat_binary_average_timeless,
    STAT_COUNT: _stat_binary_count,
    STAT_COUNT_BINARY_ON: _stat_binary_count_on,
    STAT_COUNT_BINARY_OFF: _stat_binary_count_off,
    STAT_DATETIME_NEWEST: _stat_binary_datetime_newest,
    STAT_DATETIME_OLDEST: _stat_binary_datetime_oldest,
    STAT_MEAN: _stat_binary_mean
}

STATS_NOT_A_NUMBER: Set[str] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
    STAT_DATETIME_VALUE_MIN
}

STATS_DATETIME: Set[str] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
