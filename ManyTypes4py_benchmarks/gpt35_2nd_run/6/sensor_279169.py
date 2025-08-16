from __future__ import annotations
from datetime import datetime
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_TYPE, CONF_UNIQUE_ID, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import config_validation as cv, entity_registry as er
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, StateType
from . import PLATFORMS
from .const import CONF_ENTITY_IDS, CONF_ROUND_DIGITS, DOMAIN

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_MIN_VALUE: str = 'min_value'
ATTR_MIN_ENTITY_ID: str = 'min_entity_id'
ATTR_MAX_VALUE: str = 'max_value'
ATTR_MAX_ENTITY_ID: str = 'max_entity_id'
ATTR_MEAN: str = 'mean'
ATTR_MEDIAN: str = 'median'
ATTR_LAST: str = 'last'
ATTR_LAST_ENTITY_ID: str = 'last_entity_id'
ATTR_RANGE: str = 'range'
ATTR_SUM: str = 'sum'
ICON: str = 'mdi:calculator'
SENSOR_TYPES: Dict[str, str] = {ATTR_MIN_VALUE: 'min', ATTR_MAX_VALUE: 'max', ATTR_MEAN: 'mean', ATTR_MEDIAN: 'median', ATTR_LAST: 'last', ATTR_RANGE: 'range', ATTR_SUM: 'sum'}
SENSOR_TYPE_TO_ATTR: Dict[str, str] = {v: k for k, v in SENSOR_TYPES.items()}
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({vol.Optional(CONF_TYPE, default=SENSOR_TYPES[ATTR_MAX_VALUE]): vol.All(cv.string, vol.In(SENSOR_TYPES.values())), vol.Optional(CONF_NAME): cv.string, vol.Required(CONF_ENTITY_IDS): cv.entity_ids, vol.Optional(CONF_ROUND_DIGITS, default=2): vol.Coerce(int), vol.Optional(CONF_UNIQUE_ID): cv.string}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Initialize min/max/mean config entry."""
    registry = er.async_get(hass)
    entity_ids = er.async_validate_entity_ids(registry, config_entry.options[CONF_ENTITY_IDS])
    sensor_type = config_entry.options[CONF_TYPE]
    round_digits = int(config_entry.options[CONF_ROUND_DIGITS])
    async_add_entities([MinMaxSensor(entity_ids, config_entry.title, sensor_type, round_digits, config_entry.entry_id)])

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Set up the min/max/mean sensor."""
    entity_ids = config[CONF_ENTITY_IDS]
    name = config.get(CONF_NAME)
    sensor_type = config[CONF_TYPE]
    round_digits = config[CONF_ROUND_DIGITS]
    unique_id = config.get(CONF_UNIQUE_ID)
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    async_add_entities([MinMaxSensor(entity_ids, name, sensor_type, round_digits, unique_id)]

def calc_min(sensor_values: List[Tuple[str, StateType]]) -> Tuple[Optional[str], Optional[StateType]]:
    """Calculate min value, honoring unknown states."""
    val: Optional[StateType] = None
    entity_id: Optional[str] = None
    for sensor_id, sensor_value in sensor_values:
        if sensor_value not in [STATE_UNKNOWN, STATE_UNAVAILABLE] and (val is None or val > sensor_value):
            entity_id, val = (sensor_id, sensor_value)
    return (entity_id, val)

# Define calc_max, calc_mean, calc_median, calc_range, calc_sum functions with appropriate type annotations

class MinMaxSensor(SensorEntity):
    """Representation of a min/max sensor."""
    _attr_icon: str = ICON
    _attr_should_poll: bool = False
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, entity_ids: List[str], name: Optional[str], sensor_type: str, round_digits: int, unique_id: Optional[str]) -> None:
        """Initialize the min/max sensor."""
        # Define attributes with appropriate type annotations

    async def async_added_to_hass(self) -> None:
        """Handle added to Hass."""
        # Implement async_added_to_hass method with appropriate type annotations

    @property
    def native_value(self) -> Optional[StateType]:
        """Return the state of the sensor."""
        # Implement native_value property with appropriate type annotations

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit the value is expressed in."""
        # Implement native_unit_of_measurement property with appropriate type annotations

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the state attributes of the sensor."""
        # Implement extra_state_attributes property with appropriate type annotations

    # Implement _async_min_max_sensor_state_listener, _calc_values methods with appropriate type annotations
