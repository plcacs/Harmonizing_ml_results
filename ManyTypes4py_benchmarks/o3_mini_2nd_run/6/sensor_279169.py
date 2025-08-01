from __future__ import annotations
from datetime import datetime
import logging
import statistics
from typing import Any, Optional, Tuple, List, Dict, Union
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_TYPE, CONF_UNIQUE_ID, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, StateType

from . import PLATFORMS
from .const import CONF_ENTITY_IDS, CONF_ROUND_DIGITS, DOMAIN

_LOGGER = logging.getLogger(__name__)

ATTR_MIN_VALUE = 'min_value'
ATTR_MIN_ENTITY_ID = 'min_entity_id'
ATTR_MAX_VALUE = 'max_value'
ATTR_MAX_ENTITY_ID = 'max_entity_id'
ATTR_MEAN = 'mean'
ATTR_MEDIAN = 'median'
ATTR_LAST = 'last'
ATTR_LAST_ENTITY_ID = 'last_entity_id'
ATTR_RANGE = 'range'
ATTR_SUM = 'sum'
ICON = 'mdi:calculator'

SENSOR_TYPES: Dict[str, str] = {
    ATTR_MIN_VALUE: 'min',
    ATTR_MAX_VALUE: 'max',
    ATTR_MEAN: 'mean',
    ATTR_MEDIAN: 'median',
    ATTR_LAST: 'last',
    ATTR_RANGE: 'range',
    ATTR_SUM: 'sum'
}
SENSOR_TYPE_TO_ATTR: Dict[str, str] = {v: k for k, v in SENSOR_TYPES.items()}

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_TYPE, default=SENSOR_TYPES[ATTR_MAX_VALUE]): vol.All(cv.string, vol.In(SENSOR_TYPES.values())),
    vol.Optional(CONF_NAME): cv.string,
    vol.Required(CONF_ENTITY_IDS): cv.entity_ids,
    vol.Optional(CONF_ROUND_DIGITS, default=2): vol.Coerce(int),
    vol.Optional(CONF_UNIQUE_ID): cv.string
})


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Initialize min/max/mean config entry."""
    registry = er.async_get(hass)
    entity_ids: List[str] = er.async_validate_entity_ids(registry, config_entry.options[CONF_ENTITY_IDS])
    sensor_type: str = config_entry.options[CONF_TYPE]
    round_digits: int = int(config_entry.options[CONF_ROUND_DIGITS])
    async_add_entities([
        MinMaxSensor(entity_ids, config_entry.title, sensor_type, round_digits, config_entry.entry_id)
    ])


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the min/max/mean sensor."""
    entity_ids: List[str] = config[CONF_ENTITY_IDS]
    name: Optional[str] = config.get(CONF_NAME)
    sensor_type: str = config[CONF_TYPE]
    round_digits: int = config[CONF_ROUND_DIGITS]
    unique_id: Optional[str] = config.get(CONF_UNIQUE_ID)
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    async_add_entities([
        MinMaxSensor(entity_ids, name, sensor_type, round_digits, unique_id)
    ])


def calc_min(sensor_values: List[Tuple[str, StateType]]) -> Tuple[Optional[str], Optional[float]]:
    """Calculate min value, honoring unknown states."""
    val: Optional[float] = None
    entity_id: Optional[str] = None
    for s_id, s_value in sensor_values:
        if s_value in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            continue
        if val is None or (isinstance(s_value, (int, float)) and val > s_value):
            entity_id, val = s_id, float(s_value)
    return (entity_id, val)


def calc_max(sensor_values: List[Tuple[str, StateType]]) -> Tuple[Optional[str], Optional[float]]:
    """Calculate max value, honoring unknown states."""
    val: Optional[float] = None
    entity_id: Optional[str] = None
    for s_id, s_value in sensor_values:
        if s_value in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            continue
        if val is None or (isinstance(s_value, (int, float)) and val < s_value):
            entity_id, val = s_id, float(s_value)
    return (entity_id, val)


def calc_mean(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:
    """Calculate mean value, honoring unknown states."""
    result: List[float] = [
        float(s_value) for _, s_value in sensor_values if s_value not in [STATE_UNKNOWN, STATE_UNAVAILABLE]
    ]
    if not result:
        return None
    value: float = round(statistics.mean(result), round_digits)
    return value


def calc_median(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:
    """Calculate median value, honoring unknown states."""
    result: List[float] = [
        float(s_value) for _, s_value in sensor_values if s_value not in [STATE_UNKNOWN, STATE_UNAVAILABLE]
    ]
    if not result:
        return None
    value: float = round(statistics.median(result), round_digits)
    return value


def calc_range(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:
    """Calculate range value, honoring unknown states."""
    result: List[float] = [
        float(s_value) for _, s_value in sensor_values if s_value not in [STATE_UNKNOWN, STATE_UNAVAILABLE]
    ]
    if not result:
        return None
    value: float = round(max(result) - min(result), round_digits)
    return value


def calc_sum(sensor_values: List[Tuple[str, StateType]], round_digits: int) -> Optional[float]:
    """Calculate a sum of values, not honoring unknown states."""
    total: float = 0
    for _, s_value in sensor_values:
        if s_value in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            return None
        total += float(s_value)
    value: float = round(total, round_digits)
    return value


class MinMaxSensor(SensorEntity):
    """Representation of a min/max sensor."""
    _attr_icon: str = ICON
    _attr_should_poll: bool = False
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        entity_ids: List[str],
        name: Optional[str],
        sensor_type: str,
        round_digits: int,
        unique_id: Optional[str],
    ) -> None:
        """Initialize the min/max sensor."""
        self._attr_unique_id: Optional[str] = unique_id
        self._entity_ids: List[str] = entity_ids
        self._sensor_type: str = sensor_type
        self._round_digits: int = round_digits
        if name:
            self._attr_name: str = name
        else:
            self._attr_name = f'{sensor_type} sensor'.capitalize()
        self._sensor_attr: str = SENSOR_TYPE_TO_ATTR[self._sensor_type]
        self._unit_of_measurement: Optional[str] = None
        self._unit_of_measurement_mismatch: bool = False
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None
        self.mean: Optional[float] = None
        self.last: Optional[float] = None
        self.median: Optional[float] = None
        self.range: Optional[float] = None
        self.sum: Optional[float] = None
        self.min_entity_id: Optional[str] = None
        self.max_entity_id: Optional[str] = None
        self.last_entity_id: Optional[str] = None
        self.count_sensors: int = len(self._entity_ids)
        self.states: Dict[str, Union[float, str]] = {}

    async def async_added_to_hass(self) -> None:
        """Handle added to Hass."""
        self.async_on_remove(
            async_track_state_change_event(self.hass, self._entity_ids, self._async_min_max_sensor_state_listener)
        )
        for entity_id in self._entity_ids:
            state = self.hass.states.get(entity_id)
            state_event = Event('', {'entity_id': entity_id, 'new_state': state, 'old_state': None})
            self._async_min_max_sensor_state_listener(state_event, update_state=False)
        self._calc_values()

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the sensor."""
        if self._unit_of_measurement_mismatch:
            return None
        value: Optional[float] = getattr(self, self._sensor_attr)  # type: ignore
        return value

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit the value is expressed in."""
        if self._unit_of_measurement_mismatch:
            return 'ERR'
        return self._unit_of_measurement

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the state attributes of the sensor."""
        if self._sensor_type == 'min':
            return {ATTR_MIN_ENTITY_ID: self.min_entity_id}
        if self._sensor_type == 'max':
            return {ATTR_MAX_ENTITY_ID: self.max_entity_id}
        if self._sensor_type == 'last':
            return {ATTR_LAST_ENTITY_ID: self.last_entity_id}
        return None

    @callback
    def _async_min_max_sensor_state_listener(self, event: Event, update_state: bool = True) -> None:
        """Handle the sensor state changes."""
        new_state = event.data['new_state']
        entity: str = event.data['entity_id']
        if new_state is None or new_state.state is None or new_state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
            self.states[entity] = STATE_UNKNOWN
            if not update_state:
                return
            self._calc_values()
            self.async_write_ha_state()
            return
        if self._unit_of_measurement is None:
            self._unit_of_measurement = new_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        if self._unit_of_measurement != new_state.attributes.get(ATTR_UNIT_OF_MEASUREMENT):
            _LOGGER.warning('Units of measurement do not match for entity %s', self.entity_id)
            self._unit_of_measurement_mismatch = True
        try:
            self.states[entity] = float(new_state.state)
            self.last = float(new_state.state)
            self.last_entity_id = entity
        except ValueError:
            _LOGGER.warning('Unable to store state. Only numerical states are supported')
        if not update_state:
            return
        self._calc_values()
        self.async_write_ha_state()

    @callback
    def _calc_values(self) -> None:
        """Calculate the values."""
        sensor_values: List[Tuple[str, StateType]] = [
            (e_id, self.states[e_id]) for e_id in self._entity_ids if e_id in self.states
        ]
        self.min_entity_id, self.min_value = calc_min(sensor_values)
        self.max_entity_id, self.max_value = calc_max(sensor_values)
        self.mean = calc_mean(sensor_values, self._round_digits)
        self.median = calc_median(sensor_values, self._round_digits)
        self.range = calc_range(sensor_values, self._round_digits)
        self.sum = calc_sum(sensor_values, self._round_digits)