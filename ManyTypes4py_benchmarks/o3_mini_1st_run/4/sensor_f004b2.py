"""Allows the creation of a sensor that filters state property."""
from __future__ import annotations
from collections import Counter, deque
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
import logging
from numbers import Number
import statistics
from typing import Any, Deque, Optional, Union, cast, List, Dict, Callable

import voluptuous as vol
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.input_number import DOMAIN as INPUT_NUMBER_DOMAIN
from homeassistant.components.recorder import get_instance, history
from homeassistant.components.sensor import (
    ATTR_STATE_CLASS,
    DOMAIN as SENSOR_DOMAIN,
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_ENTITY_ID,
    ATTR_ICON,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_UNIQUE_ID,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, StateType
from homeassistant.util import dt as dt_util
from homeassistant.util.decorator import Registry
from .const import (
    CONF_FILTER_LOWER_BOUND,
    CONF_FILTER_NAME,
    CONF_FILTER_PRECISION,
    CONF_FILTER_RADIUS,
    CONF_FILTER_TIME_CONSTANT,
    CONF_FILTER_UPPER_BOUND,
    CONF_FILTER_WINDOW_SIZE,
    CONF_FILTERS,
    CONF_TIME_SMA_TYPE,
    DEFAULT_FILTER_RADIUS,
    DEFAULT_FILTER_TIME_CONSTANT,
    DEFAULT_PRECISION,
    DEFAULT_WINDOW_SIZE,
    DOMAIN,
    FILTER_NAME_LOWPASS,
    FILTER_NAME_OUTLIER,
    FILTER_NAME_RANGE,
    FILTER_NAME_THROTTLE,
    FILTER_NAME_TIME_SMA,
    FILTER_NAME_TIME_THROTTLE,
    PLATFORMS,
    TIME_SMA_LAST,
    WINDOW_SIZE_UNIT_NUMBER_EVENTS,
    WINDOW_SIZE_UNIT_TIME,
)

_LOGGER = logging.getLogger(__name__)
FILTERS = Registry()
ICON: str = 'mdi:chart-line-variant'

FILTER_SCHEMA = vol.Schema({vol.Optional(CONF_FILTER_PRECISION): vol.Coerce(int)})

FILTER_OUTLIER_SCHEMA = FILTER_SCHEMA.extend({
    vol.Required(CONF_FILTER_NAME): FILTER_NAME_OUTLIER,
    vol.Optional(CONF_FILTER_WINDOW_SIZE, default=DEFAULT_WINDOW_SIZE): vol.Coerce(int),
    vol.Optional(CONF_FILTER_RADIUS, default=DEFAULT_FILTER_RADIUS): vol.Coerce(float)
})

FILTER_LOWPASS_SCHEMA = FILTER_SCHEMA.extend({
    vol.Required(CONF_FILTER_NAME): FILTER_NAME_LOWPASS,
    vol.Optional(CONF_FILTER_WINDOW_SIZE, default=DEFAULT_WINDOW_SIZE): vol.Coerce(int),
    vol.Optional(CONF_FILTER_TIME_CONSTANT, default=DEFAULT_FILTER_TIME_CONSTANT): vol.Coerce(int)
})

FILTER_RANGE_SCHEMA = FILTER_SCHEMA.extend({
    vol.Required(CONF_FILTER_NAME): FILTER_NAME_RANGE,
    vol.Optional(CONF_FILTER_LOWER_BOUND): vol.Coerce(float),
    vol.Optional(CONF_FILTER_UPPER_BOUND): vol.Coerce(float)
})

FILTER_TIME_SMA_SCHEMA = FILTER_SCHEMA.extend({
    vol.Required(CONF_FILTER_NAME): FILTER_NAME_TIME_SMA,
    vol.Optional(CONF_TIME_SMA_TYPE, default=TIME_SMA_LAST): vol.In([TIME_SMA_LAST]),
    vol.Required(CONF_FILTER_WINDOW_SIZE): vol.All(cv.time_period, cv.positive_timedelta)
})

FILTER_THROTTLE_SCHEMA = FILTER_SCHEMA.extend({
    vol.Required(CONF_FILTER_NAME): FILTER_NAME_THROTTLE,
    vol.Optional(CONF_FILTER_WINDOW_SIZE, default=DEFAULT_WINDOW_SIZE): vol.Coerce(int)
})

FILTER_TIME_THROTTLE_SCHEMA = FILTER_SCHEMA.extend({
    vol.Required(CONF_FILTER_NAME): FILTER_NAME_TIME_THROTTLE,
    vol.Required(CONF_FILTER_WINDOW_SIZE): vol.All(cv.time_period, cv.positive_timedelta)
})

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_ENTITY_ID): vol.Any(cv.entity_domain(SENSOR_DOMAIN), cv.entity_domain(BINARY_SENSOR_DOMAIN), cv.entity_domain(INPUT_NUMBER_DOMAIN)),
    vol.Optional(CONF_NAME): cv.string,
    vol.Optional(CONF_UNIQUE_ID): cv.string,
    vol.Required(CONF_FILTERS): vol.All(cv.ensure_list, [vol.Any(
        FILTER_OUTLIER_SCHEMA,
        FILTER_LOWPASS_SCHEMA,
        FILTER_TIME_SMA_SCHEMA,
        FILTER_THROTTLE_SCHEMA,
        FILTER_TIME_THROTTLE_SCHEMA,
        FILTER_RANGE_SCHEMA
    )])
})


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the template sensors."""
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    name: Optional[str] = config.get(CONF_NAME)
    unique_id: Optional[str] = config.get(CONF_UNIQUE_ID)
    entity_id: str = config[CONF_ENTITY_ID]
    filter_configs: List[Dict[str, Any]] = config[CONF_FILTERS]
    filters: List[Filter] = [FILTERS[_filter.pop(CONF_FILTER_NAME)](entity=entity_id, **_filter) for _filter in filter_configs]
    async_add_entities([SensorFilter(name, unique_id, entity_id, filters)])


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Filter sensor entry."""
    name: str = entry.options[CONF_NAME]
    entity_id: str = entry.options[CONF_ENTITY_ID]
    filter_config: Dict[str, Any] = {k: v for k, v in entry.options.items() if k not in (CONF_NAME, CONF_ENTITY_ID)}
    if CONF_FILTER_WINDOW_SIZE in filter_config and isinstance(filter_config[CONF_FILTER_WINDOW_SIZE], dict):
        filter_config[CONF_FILTER_WINDOW_SIZE] = timedelta(**filter_config[CONF_FILTER_WINDOW_SIZE])
    filters: List[Filter] = [FILTERS[filter_config.pop(CONF_FILTER_NAME)](entity=entity_id, **filter_config)]
    async_add_entities([SensorFilter(name, entry.entry_id, entity_id, filters)])


class SensorFilter(SensorEntity):
    """Representation of a Filter Sensor."""
    _attr_should_poll: bool = False

    def __init__(self, name: Optional[str], unique_id: Optional[str], entity_id: str, filters: List[Filter]) -> None:
        """Initialize the sensor."""
        self._attr_name: Optional[str] = name
        self._attr_unique_id: Optional[str] = unique_id
        self._entity: str = entity_id
        self._attr_native_unit_of_measurement: Optional[str] = None
        self._state: Optional[Any] = None
        self._filters: List[Filter] = filters
        self._attr_icon: Optional[str] = None
        self._attr_device_class: Optional[str] = None
        self._attr_state_class: Optional[str] = None
        self._attr_extra_state_attributes: Dict[str, Any] = {ATTR_ENTITY_ID: entity_id}

    @callback
    def _update_filter_sensor_state_event(self, event: Event) -> None:
        """Handle device state changes."""
        _LOGGER.debug('Update filter on event: %s', event)
        self._update_filter_sensor_state(event.data['new_state'])

    @callback
    def _update_filter_sensor_state(self, new_state: Optional[State], update_ha: bool = True) -> None:
        """Process device state changes."""
        if new_state is None:
            _LOGGER.warning('While updating filter %s, the new_state is None', self._attr_name)
            self._state = None
            self.async_write_ha_state()
            return
        if new_state.state == STATE_UNKNOWN:
            self._state = None
            self.async_write_ha_state()
            return
        if new_state.state == STATE_UNAVAILABLE:
            self._attr_available = False
            self.async_write_ha_state()
            return
        self._attr_available = True
        temp_state = _State(new_state.last_updated, new_state.state)
        try:
            for filt in self._filters:
                filtered_state: State = filt.filter_state(copy(temp_state))
                _LOGGER.debug('%s(%s=%s) -> %s', filt.name, self._entity, temp_state.state, 'skip' if filt.skip_processing else filtered_state.state)
                if filt.skip_processing:
                    return
                temp_state = filtered_state
        except ValueError:
            _LOGGER.error('Could not convert state: %s (%s) to number', new_state.state, type(new_state.state))
            return
        self._state = temp_state.state
        self._attr_icon = new_state.attributes.get(ATTR_ICON, ICON)
        self._attr_device_class = new_state.attributes.get(ATTR_DEVICE_CLASS)
        self._attr_state_class = new_state.attributes.get(ATTR_STATE_CLASS)
        if self._attr_native_unit_of_measurement != new_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT):
            for filt in self._filters:
                filt.reset()
            self._attr_native_unit_of_measurement = new_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        if update_ha:
            self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        if 'recorder' in self.hass.config.components:
            history_list: List[State] = []
            largest_window_items: int = 0
            largest_window_time: timedelta = timedelta(0)
            for filt in self._filters:
                if filt.window_unit == WINDOW_SIZE_UNIT_NUMBER_EVENTS and largest_window_items < (size := cast(int, filt.window_size)):
                    largest_window_items = size
                elif filt.window_unit == WINDOW_SIZE_UNIT_TIME and largest_window_time < (val := cast(timedelta, filt.window_size)):
                    largest_window_time = val
            if largest_window_items > 0:
                filter_history: Dict[str, List[State]] = await get_instance(self.hass).async_add_executor_job(
                    partial(history.get_last_state_changes, self.hass, largest_window_items, entity_id=self._entity)
                )
                if self._entity in filter_history:
                    history_list.extend(filter_history[self._entity])
            if largest_window_time > timedelta(seconds=0):
                start: datetime = dt_util.utcnow() - largest_window_time
                filter_history = await get_instance(self.hass).async_add_executor_job(
                    partial(history.state_changes_during_period, self.hass, start, entity_id=self._entity)
                )
                if self._entity in filter_history:
                    history_list.extend([state for state in filter_history[self._entity] if state not in history_list])
            history_list = sorted(history_list, key=lambda s: s.last_updated)
            _LOGGER.debug('Loading from history: %s', [(s.state, s.last_updated) for s in history_list])
            for state in history_list:
                if state.state not in [STATE_UNKNOWN, STATE_UNAVAILABLE, None]:
                    self._update_filter_sensor_state(state, False)

        @callback
        def _async_hass_started(hass: HomeAssistant) -> None:
            """Delay source entity tracking."""
            self.async_on_remove(async_track_state_change_event(self.hass, [self._entity], self._update_filter_sensor_state_event))
        self.async_on_remove(async_at_started(self.hass, _async_hass_started))

    @property
    def native_value(self) -> Optional[StateType]:
        """Return the state of the sensor."""
        if self._state is not None and self.device_class == SensorDeviceClass.TIMESTAMP:
            return datetime.fromisoformat(str(self._state))
        return self._state


class FilterState:
    """State abstraction for filter usage."""

    def __init__(self, state: State) -> None:
        """Initialize with HA State object."""
        self.timestamp: datetime = state.last_updated
        try:
            self.state: Union[float, str] = float(state.state)
        except ValueError:
            self.state = state.state

    def set_precision(self, precision: Optional[int]) -> None:
        """Set precision of Number based states."""
        if precision is not None and isinstance(self.state, Number):
            value: float = round(float(self.state), precision)
            self.state = int(value) if precision == 0 else value

    def __str__(self) -> str:
        """Return state as the string representation of FilterState."""
        return str(self.state)

    def __repr__(self) -> str:
        """Return timestamp and state as the representation of FilterState."""
        return f'{self.timestamp} : {self.state}'


@dataclass
class _State:
    """Simplified State class.

    The standard State class only accepts string in `state`,
    and we are only interested in two properties.
    """
    timestamp: datetime
    state: Union[float, str]


class Filter:
    """Filter skeleton."""

    def __init__(self, name: str, window_size: Union[int, timedelta], entity: str, precision: Optional[int]) -> None:
        """Initialize common attributes.

        :param window_size: size of the sliding window that holds previous values
        :param precision: round filtered value to precision value
        :param entity: used for debugging only
        """
        if isinstance(window_size, int):
            self.states: Deque[FilterState] = deque(maxlen=window_size)
            self.window_unit: str = WINDOW_SIZE_UNIT_NUMBER_EVENTS
        else:
            self.states = deque(maxlen=0)
            self.window_unit = WINDOW_SIZE_UNIT_TIME
        self.filter_precision: Optional[int] = precision
        self._name: str = name
        self._entity: str = entity
        self._skip_processing: bool = False
        self._window_size: Union[int, timedelta] = window_size
        self._store_raw: bool = False
        self._only_numbers: bool = True

    @property
    def window_size(self) -> Union[int, timedelta]:
        """Return window size."""
        return self._window_size

    @property
    def name(self) -> str:
        """Return filter name."""
        return self._name

    @property
    def skip_processing(self) -> bool:
        """Return whether the current filter_state should be skipped."""
        return self._skip_processing

    def reset(self) -> None:
        """Reset filter."""
        self.states.clear()

    def _filter_state(self, new_state: FilterState) -> FilterState:
        """Implement filter."""
        raise NotImplementedError

    def filter_state(self, new_state: State) -> State:
        """Implement a common interface for filters."""
        fstate = FilterState(new_state)
        if self._only_numbers and (not isinstance(fstate.state, Number)):
            raise ValueError(f'State <{fstate.state}> is not a Number')
        filtered: FilterState = self._filter_state(fstate)
        filtered.set_precision(self.filter_precision)
        if self._store_raw:
            self.states.append(copy(FilterState(new_state)))
        else:
            self.states.append(copy(filtered))
        new_state.state = filtered.state
        return new_state


@FILTERS.register(FILTER_NAME_RANGE)
class RangeFilter(Filter, SensorEntity):
    """Range filter.

    Determines if new state is in the range of upper_bound and lower_bound.
    If not inside, lower or upper bound is returned instead.
    """

    def __init__(self, *, entity: str, precision: Optional[int] = None, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None) -> None:
        """Initialize Filter.

        :param upper_bound: band upper bound
        :param lower_bound: band lower bound
        """
        super().__init__(FILTER_NAME_RANGE, DEFAULT_WINDOW_SIZE, entity=entity, precision=precision)
        self._lower_bound: Optional[float] = lower_bound
        self._upper_bound: Optional[float] = upper_bound
        self._stats_internal: Counter[str] = Counter()

    def _filter_state(self, new_state: FilterState) -> FilterState:
        """Implement the range filter."""
        new_state_value: float = cast(float, new_state.state)
        if self._upper_bound is not None and new_state_value > self._upper_bound:
            self._stats_internal['erasures_up'] += 1
            _LOGGER.debug('Upper outlier nr. %s in %s: %s', self._stats_internal['erasures_up'], self._entity, new_state)
            new_state.state = self._upper_bound
        elif self._lower_bound is not None and new_state_value < self._lower_bound:
            self._stats_internal['erasures_low'] += 1
            _LOGGER.debug('Lower outlier nr. %s in %s: %s', self._stats_internal['erasures_low'], self._entity, new_state)
            new_state.state = self._lower_bound
        return new_state


@FILTERS.register(FILTER_NAME_OUTLIER)
class OutlierFilter(Filter, SensorEntity):
    """BASIC outlier filter.

    Determines if new state is in a band around the median.
    """

    def __init__(self, *, window_size: int, entity: str, radius: float, precision: Optional[int] = None) -> None:
        """Initialize Filter.

        :param radius: band radius
        """
        super().__init__(FILTER_NAME_OUTLIER, window_size, entity=entity, precision=precision)
        self._radius: float = radius
        self._stats_internal: Counter[str] = Counter()
        self._store_raw = True

    def _filter_state(self, new_state: FilterState) -> FilterState:
        """Implement the outlier filter."""
        previous_state_values: List[float] = [cast(float, s.state) for s in self.states]
        new_state_value: float = cast(float, new_state.state)
        median: float = statistics.median(previous_state_values) if self.states else 0
        if len(self.states) == self.states.maxlen and abs(new_state_value - median) > self._radius:
            self._stats_internal['erasures'] += 1
            _LOGGER.debug('Outlier nr. %s in %s: %s', self._stats_internal['erasures'], self._entity, new_state)
            new_state.state = median
        return new_state


@FILTERS.register(FILTER_NAME_LOWPASS)
class LowPassFilter(Filter, SensorEntity):
    """BASIC Low Pass Filter."""

    def __init__(self, *, window_size: int, entity: str, time_constant: int, precision: int = DEFAULT_PRECISION) -> None:
        """Initialize Filter."""
        super().__init__(FILTER_NAME_LOWPASS, window_size, entity=entity, precision=precision)
        self._time_constant: int = time_constant

    def _filter_state(self, new_state: FilterState) -> FilterState:
        """Implement the low pass filter."""
        if not self.states:
            return new_state
        new_weight: float = 1.0 / self._time_constant
        prev_weight: float = 1.0 - new_weight
        prev_state_value: float = cast(float, self.states[-1].state)
        new_state_value: float = cast(float, new_state.state)
        new_state.state = prev_weight * prev_state_value + new_weight * new_state_value
        return new_state


@FILTERS.register(FILTER_NAME_TIME_SMA)
class TimeSMAFilter(Filter, SensorEntity):
    """Simple Moving Average (SMA) Filter.

    The window_size is determined by time, and SMA is time weighted.
    """

    def __init__(self, *, window_size: timedelta, entity: str, type: str, precision: int = DEFAULT_PRECISION) -> None:
        """Initialize Filter.

        :param type: type of algorithm used to connect discrete values
        """
        super().__init__(FILTER_NAME_TIME_SMA, window_size, entity=entity, precision=precision)
        self._time_window: timedelta = window_size
        self.last_leak: Optional[FilterState] = None
        self.queue: Deque[FilterState] = deque()

    def _leak(self, left_boundary: datetime) -> None:
        """Remove timeouted elements."""
        while self.queue:
            if self.queue[0].timestamp + self._time_window <= left_boundary:
                self.last_leak = self.queue.popleft()
            else:
                return

    def _filter_state(self, new_state: FilterState) -> FilterState:
        """Implement the Simple Moving Average filter."""
        self._leak(new_state.timestamp)
        self.queue.append(copy(new_state))
        moving_sum: float = 0
        start: datetime = new_state.timestamp - self._time_window
        prev_state: FilterState = self.last_leak if self.last_leak is not None else self.queue[0]
        for state in self.queue:
            prev_state_value: float = cast(float, prev_state.state)
            moving_sum += (state.timestamp - start).total_seconds() * prev_state_value
            start = state.timestamp
            prev_state = state
        new_state.state = moving_sum / self._time_window.total_seconds()
        return new_state


@FILTERS.register(FILTER_NAME_THROTTLE)
class ThrottleFilter(Filter, SensorEntity):
    """Throttle Filter.

    One sample per window.
    """

    def __init__(self, *, window_size: int, entity: str, precision: Optional[int] = None) -> None:
        """Initialize Filter."""
        super().__init__(FILTER_NAME_THROTTLE, window_size, entity=entity, precision=precision)
        self._only_numbers = False

    def _filter_state(self, new_state: FilterState) -> FilterState:
        """Implement the throttle filter."""
        if not self.states or len(self.states) == self.states.maxlen:
            self.states.clear()
            self._skip_processing = False
        else:
            self._skip_processing = True
        return new_state


@FILTERS.register(FILTER_NAME_TIME_THROTTLE)
class TimeThrottleFilter(Filter, SensorEntity):
    """Time Throttle Filter.

    One sample per time period.
    """

    def __init__(self, *, window_size: timedelta, entity: str, precision: Optional[int] = None) -> None:
        """Initialize Filter."""
        super().__init__(FILTER_NAME_TIME_THROTTLE, window_size, entity=entity, precision=precision)
        self._time_window: timedelta = window_size
        self._last_emitted_at: Optional[datetime] = None
        self._only_numbers = False

    def _filter_state(self, new_state: FilterState) -> FilterState:
        """Implement the filter."""
        window_start: datetime = new_state.timestamp - self._time_window
        if not self._last_emitted_at or self._last_emitted_at <= window_start:
            self._last_emitted_at = new_state.timestamp
            self._skip_processing = False
        else:
            self._skip_processing = True
        return new_state
