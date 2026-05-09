from __future__ import annotations
from collections import deque
from collections.abc import Callable, Mapping
import contextlib
from datetime import datetime, timedelta
import logging
import math
import statistics
import time
from typing import Any, cast
import voluptuous as vol
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.recorder import get_instance, history
from homeassistant.components.sensor import DEVICE_CLASS_STATE_CLASSES, DEVICE_CLASS_UNITS, PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_DEVICE_CLASS, ATTR_UNIT_OF_MEASUREMENT, CONF_ENTITY_ID, CONF_NAME, CONF_UNIQUE_ID, PERCENTAGE, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, EventStateReportedData, HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.event import async_track_point_in_utc_time, async_track_state_change_event, async_track_state_report_event
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.util import dt as dt_util
from homeassistant.util.enum import try_parse_enum
from . import DOMAIN, PLATFORMS

_LOGGER: logging.Logger = logging.getLogger(__name__)

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

STATS_NUMERIC_SUPPORT: Mapping[str, Callable[[deque[bool | float], deque[datetime], int], datetime | int | float | None]] = {
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

STATS_BINARY_SUPPORT: Mapping[str, Callable[[deque[bool], deque[datetime], int], datetime | int | float | None]] = {
    STAT_AVERAGE_STEP: _stat_binary_average_step,
    STAT_AVERAGE_TIMELESS: _stat_binary_average_timeless,
    STAT_COUNT: _stat_binary_count,
    STAT_COUNT_BINARY_ON: _stat_binary_count_on,
    STAT_COUNT_BINARY_OFF: _stat_binary_count_off,
    STAT_DATETIME_NEWEST: _stat_binary_datetime_newest,
    STAT_DATETIME_OLDEST: _stat_binary_datetime_oldest,
    STAT_MEAN: _stat_binary_mean
}

STATS_NOT_A_NUMBER: Mapping[str, None] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
    STAT_DATETIME_VALUE_MIN
}

STATS_DATETIME: Mapping[str, None] = {
    STAT_DATETIME_NEWEST,
    STAT_DATETIME_OLDEST,
    STAT_DATETIME_VALUE_MAX,
    STAT_DATETIME_VALUE_MIN
}

STATS_NUMERIC_RETAIN_UNIT: Mapping[str, None] = {
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
    STAT_VARIANCE
}

STATS_BINARY_PERCENTAGE: Mapping[str, None] = {
    STAT_AVERAGE_STEP,
    STAT_AVERAGE_TIMELESS,
    STAT_MEAN
}

CONF_STATE_CHARACTERISTIC: str = 'state_characteristic'
CONF_SAMPLES_MAX_BUFFER_SIZE: str = 'sampling_size'
CONF_MAX_AGE: str = 'max_age'
CONF_KEEP_LAST_SAMPLE: str = 'keep_last_sample'
CONF_PRECISION: str = 'precision'
CONF_PERCENTILE: str = 'percentile'
DEFAULT_NAME: str = 'Statistical characteristic'
DEFAULT_PRECISION: int = 2
ICON: str = 'mdi