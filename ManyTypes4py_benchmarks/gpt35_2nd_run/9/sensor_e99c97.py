from __future__ import annotations
from collections.abc import Callable
from datetime import datetime
import logging
import statistics
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import voluptuous as vol
from homeassistant.components.input_number import DOMAIN as INPUT_NUMBER_DOMAIN
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.sensor import CONF_STATE_CLASS, DEVICE_CLASS_UNITS, DEVICE_CLASSES_SCHEMA, DOMAIN as SENSOR_DOMAIN, PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, STATE_CLASSES_SCHEMA, UNIT_CONVERTERS, SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_DEVICE_CLASS, CONF_ENTITIES, CONF_NAME, CONF_TYPE, CONF_UNIQUE_ID, CONF_UNIT_OF_MEASUREMENT, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_registry as er
from homeassistant.helpers.entity import get_capability, get_device_class, get_unit_of_measurement
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue, async_delete_issue
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, StateType
from .const import CONF_IGNORE_NON_NUMERIC, DOMAIN as GROUP_DOMAIN
from .entity import GroupEntity

DEFAULT_NAME: str = 'Sensor Group'
ATTR_MIN_VALUE: str = 'min_value'
ATTR_MIN_ENTITY_ID: str = 'min_entity_id'
ATTR_MAX_VALUE: str = 'max_value'
ATTR_MAX_ENTITY_ID: str = 'max_entity_id'
ATTR_MEAN: str = 'mean'
ATTR_MEDIAN: str = 'median'
ATTR_LAST: str = 'last'
ATTR_LAST_ENTITY_ID: str = 'last_entity_id'
ATTR_RANGE: str = 'range'
ATTR_STDEV: str = 'stdev'
ATTR_SUM: str = 'sum'
ATTR_PRODUCT: str = 'product'
SENSOR_TYPES: Dict[str, str] = {ATTR_MIN_VALUE: 'min', ATTR_MAX_VALUE: 'max', ATTR_MEAN: 'mean', ATTR_MEDIAN: 'median', ATTR_LAST: 'last', ATTR_RANGE: 'range', ATTR_STDEV: 'stdev', ATTR_SUM: 'sum', ATTR_PRODUCT: 'product'}
SENSOR_TYPE_TO_ATTR: Dict[str, str] = {v: k for k, v in SENSOR_TYPES.items()}
PARALLEL_UPDATES: int = 0

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Set up the Switch Group platform."""
    async_add_entities([SensorGroup(hass, config.get(CONF_UNIQUE_ID), config[CONF_NAME], config[CONF_ENTITIES], config[CONF_IGNORE_NON_NUMERIC], config[CONF_TYPE], config.get(CONF_UNIT_OF_MEASUREMENT), config.get(CONF_STATE_CLASS), config.get(CONF_DEVICE_CLASS)])

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    """Initialize Switch Group config entry."""
    registry = er.async_get(hass)
    entities = er.async_validate_entity_ids(registry, config_entry.options[CONF_ENTITIES])
    async_add_entities([SensorGroup(hass, config_entry.entry_id, config_entry.title, entities, config_entry.options.get(CONF_IGNORE_NON_NUMERIC, True), config_entry.options[CONF_TYPE], None, None, None])

@callback
def async_create_preview_sensor(hass: HomeAssistant, name: str, validated_config: ConfigType) -> SensorGroup:
    """Create a preview sensor."""
    return SensorGroup(hass, None, name, validated_config[CONF_ENTITIES], validated_config.get(CONF_IGNORE_NON_NUMERIC, False), validated_config[CONF_TYPE], None, None, None)

def _has_numeric_state(hass: HomeAssistant, entity_id: str) -> bool:
    """Test if state is numeric."""
    if not (state := hass.states.get(entity_id)):
        return False
    try:
        float(state.state)
    except ValueError:
        return False
    return True

def calc_min(sensor_values: List[Tuple[str, float, State]]) -> Tuple[Dict[str, str], float]:
    """Calculate min value."""
    val = None
    entity_id = None
    for sensor_id, sensor_value, _ in sensor_values:
        if val is None or val > sensor_value:
            entity_id, val = (sensor_id, sensor_value)
    attributes = {ATTR_MIN_ENTITY_ID: entity_id}
    if TYPE_CHECKING:
        assert val is not None
    return (attributes, val)

# Define other calc functions with appropriate type annotations

class SensorGroup(GroupEntity, SensorEntity):
    """Representation of a sensor group."""
    _attr_available: bool = False
    _attr_should_poll: bool = False

    def __init__(self, hass: HomeAssistant, unique_id: str, name: str, entity_ids: List[str], ignore_non_numeric: bool, sensor_type: str, unit_of_measurement: str, state_class: str, device_class: str) -> None:
        """Initialize a sensor group."""
        # Implementation of __init__ method

    def calculate_state_attributes(self, valid_state_entities: List[str]) -> None:
        """Calculate state attributes."""
        # Implementation of calculate_state_attributes method

    @callback
    def async_update_group_state(self) -> None:
        """Query all members and determine the sensor group state."""
        # Implementation of async_update_group_state method

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the sensor."""
        # Implementation of extra_state_attributes property

    @property
    def icon(self) -> str:
        """Return the icon."""
        # Implementation of icon property

    def _calculate_state_class(self, state_class: str, valid_state_entities: List[str]) -> str:
        """Calculate state class."""
        # Implementation of _calculate_state_class method

    def _calculate_device_class(self, device_class: str, valid_state_entities: List[str]) -> str:
        """Calculate device class."""
        # Implementation of _calculate_device_class method

    def _calculate_unit_of_measurement(self, unit_of_measurement: str, valid_state_entities: List[str]) -> str:
        """Calculate the unit of measurement."""
        # Implementation of _calculate_unit_of_measurement method

    def _get_valid_units(self) -> set:
        """Return valid units."""
        # Implementation of _get_valid_units method

    def _get_valid_entities(self) -> List[str]:
        """Return list of valid entities."""
        # Implementation of _get_valid_entities method
