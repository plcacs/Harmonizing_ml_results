"""Platform allowing several sensors to be grouped into one sensor to provide numeric combinations."""
from __future__ import annotations
from collections.abc import Callable
from datetime import datetime
import logging
import statistics
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, TypeAlias, TypeGuard, TypeVar
import voluptuous as vol
from homeassistant.components.input_number import DOMAIN as INPUT_NUMBER_DOMAIN
from homeassistant.components.number import DOMAIN as NUMBER_DOMAIN
from homeassistant.components.sensor import (
    CONF_STATE_CLASS,
    DEVICE_CLASS_UNITS,
    DEVICE_CLASSES_SCHEMA,
    DOMAIN as SENSOR_DOMAIN,
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    STATE_CLASSES_SCHEMA,
    UNIT_CONVERTERS,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_DEVICE_CLASS,
    CONF_ENTITIES,
    CONF_NAME,
    CONF_TYPE,
    CONF_UNIQUE_ID,
    CONF_UNIT_OF_MEASUREMENT,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_registry as er
from homeassistant.helpers.entity import (
    get_capability,
    get_device_class,
    get_unit_of_measurement,
)
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
    AddEntitiesCallback,
)
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
SENSOR_TYPES: Dict[str, str] = {
    ATTR_MIN_VALUE: 'min',
    ATTR_MAX_VALUE: 'max',
    ATTR_MEAN: 'mean',
    ATTR_MEDIAN: 'median',
    ATTR_LAST: 'last',
    ATTR_RANGE: 'range',
    ATTR_STDEV: 'stdev',
    ATTR_SUM: 'sum',
    ATTR_PRODUCT: 'product',
}
SENSOR_TYPE_TO_ATTR: Dict[str, str] = {v: k for k, v in SENSOR_TYPES.items()}

PARALLEL_UPDATES: int = 0
PLATFORM_SCHEMA: vol.Schema = (
    SENSOR_PLATFORM_SCHEMA
    .extend(
        {
            vol.Required(CONF_ENTITIES): cv.entities_domain([SENSOR_DOMAIN, NUMBER_DOMAIN, INPUT_NUMBER_DOMAIN]),
            vol.Required(CONF_TYPE): vol.All(cv.string, vol.In(SENSOR_TYPES.values())),
            vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
            vol.Optional(CONF_UNIQUE_ID): cv.string,
            vol.Optional(CONF_IGNORE_NON_NUMERIC, default=False): cv.boolean,
            vol.Optional(CONF_UNIT_OF_MEASUREMENT): str,
            vol.Optional(CONF_DEVICE_CLASS): DEVICE_CLASSES_SCHEMA,
            vol.Optional(CONF_STATE_CLASS): STATE_CLASSES_SCHEMA,
        }
    )
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Switch Group platform."""
    async_add_entities([SensorGroup(hass, config.get(CONF_UNIQUE_ID), config[CONF_NAME], config[CONF_ENTITIES], config[CONF_IGNORE_NON_NUMERIC], config[CONF_TYPE], config.get(CONF_UNIT_OF_MEASUREMENT), config.get(CONF_STATE_CLASS), config.get(CONF_DEVICE_CLASS))])

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Initialize Switch Group config entry."""
    registry = er.async_get(hass)
    entities = er.async_validate_entity_ids(registry, config_entry.options[CONF_ENTITIES])
    async_add_entities([SensorGroup(hass, config_entry.entry_id, config_entry.title, entities, config_entry.options.get(CONF_IGNORE_NON_NUMERIC, True), config_entry.options[CONF_TYPE], None, None, None)])

@callback
def async_create_preview_sensor(
    hass: HomeAssistant,
    name: str,
    validated_config: ConfigType,
) -> SensorGroup:
    """Create a preview sensor."""
    return SensorGroup(hass, None, name, validated_config[CONF_ENTITIES], validated_config.get(CONF_IGNORE_NON_NUMERIC, False), validated_config[CONF_TYPE], None, None, None)

def _has_numeric_state(
    hass: HomeAssistant,
    entity_id: str,
) -> TypeGuard[State]:
    """Test if state is numeric."""
    if not (state := hass.states.get(entity_id)):
        return False
    try:
        float(state.state)
    except ValueError:
        return False
    return True

def calc_min(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate min value."""
    val: Optional[float] = None
    entity_id: Optional[str] = None
    for sensor_id, sensor_value, _ in sensor_values:
        if val is None or val > sensor_value:
            entity_id, val = (sensor_id, sensor_value)
    attributes: Dict[str, str] = {ATTR_MIN_ENTITY_ID: entity_id}
    if TYPE_CHECKING:
        assert val is not None
    return (attributes, val)

def calc_max(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate max value."""
    val: Optional[float] = None
    entity_id: Optional[str] = None
    for sensor_id, sensor_value, _ in sensor_values:
        if val is None or val < sensor_value:
            entity_id, val = (sensor_id, sensor_value)
    attributes: Dict[str, str] = {ATTR_MAX_ENTITY_ID: entity_id}
    if TYPE_CHECKING:
        assert val is not None
    return (attributes, val)

def calc_mean(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate mean value."""
    result: Iterable[float] = (sensor_value for _, sensor_value, _ in sensor_values)
    value: float = statistics.mean(result)
    return ({}, value)

def calc_median(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate median value."""
    result: Iterable[float] = (sensor_value for _, sensor_value, _ in sensor_values)
    value: float = statistics.median(result)
    return ({}, value)

def calc_last(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate last value."""
    last_updated: Optional[datetime] = None
    last_entity_id: Optional[str] = None
    last: Optional[float] = None
    for entity_id, state_f, state in sensor_values:
        if last_updated is None or state.last_updated > last_updated:
            last_updated = state.last_updated
            last = state_f
            last_entity_id = entity_id
    attributes: Dict[str, str] = {ATTR_LAST_ENTITY_ID: last_entity_id}
    return (attributes, last)

def calc_range(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate range value."""
    max_result: float = max((sensor_value for _, sensor_value, _ in sensor_values))
    min_result: float = min((sensor_value for _, sensor_value, _ in sensor_values))
    value: float = max_result - min_result
    return ({}, value)

def calc_stdev(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate standard deviation value."""
    result: Iterable[float] = (sensor_value for _, sensor_value, _ in sensor_values)
    value: float = statistics.stdev(result)
    return ({}, value)

def calc_sum(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate a sum of values."""
    result: float = 0.0
    for _, sensor_value, _ in sensor_values:
        result += sensor_value
    return ({}, result)

def calc_product(
    sensor_values: List[Tuple[str, float, State]],
) -> Tuple[Dict[str, str], float]:
    """Calculate a product of values."""
    result: float = 1.0
    for _, sensor_value, _ in sensor_values:
        result *= sensor_value
    return ({}, result)

CALC_TYPES: Dict[str, Callable[[List[Tuple[str, float, State]]], Tuple[Dict[str, str], float]]] = {
    'min': calc_min,
    'max': calc_max,
    'mean': calc_mean,
    'median': calc_median,
    'last': calc_last,
    'range': calc_range,
    'stdev': calc_stdev,
    'sum': calc_sum,
    'product': calc_product,
}

class SensorGroup(GroupEntity, SensorEntity):
    """Representation of a sensor group."""

    _attr_available: bool = False
    _attr_should_poll: bool = False

    def __init__(
        self,
        hass: HomeAssistant,
        unique_id: Optional[str],
        name: str,
        entity_ids: List[str],
        ignore_non_numeric: bool,
        sensor_type: str,
        unit_of_measurement: Optional[str],
        state_class: Optional[str],
        device_class: Optional[str],
    ) -> None:
        """Initialize a sensor group."""
        self.hass = hass
        self._entity_ids: List[str] = entity_ids
        self._sensor_type: str = sensor_type
        self._configured_state_class: Optional[str] = state_class
        self._configured_device_class: Optional[str] = device_class
        self._configured_unit_of_measurement: Optional[str] = unit_of_measurement
        self._valid_units: Set[str] = set()
        self._can_convert: bool = False
        self._attr_name: str = name
        if name == DEFAULT_NAME:
            self._attr_name = f'{DEFAULT_NAME} {sensor_type}'.capitalize()
        self._attr_extra_state_attributes: Dict[str, str] = {ATTR_ENTITY_ID: entity_ids}
        self._attr_unique_id: Optional[str] = unique_id
        self._ignore_non_numeric: bool = ignore_non_numeric
        self.mode: Callable[[Iterable[bool]], bool] = all if ignore_non_numeric is False else any
        self._state_calc: Callable[[List[Tuple[str, float, State]]], Tuple[Dict[str, str], float]] = CALC_TYPES[self._sensor_type]
        self._state_incorrect: Set[str] = set()
        self._extra_state_attribute: Dict[str, str] = {}

    def calculate_state_attributes(
        self,
        valid_state_entities: List[str],
    ) -> None:
        """Calculate state attributes."""
        self._attr_state_class: Optional[str] = self._calculate_state_class(self._configured_state_class, valid_state_entities)
        self._attr_device_class: Optional[str] = self._calculate_device_class(self._configured_device_class, valid_state_entities)
        self._attr_native_unit_of_measurement: Optional[str] = self._calculate_unit_of_measurement(self._configured_unit_of_measurement, valid_state_entities)
        self._valid_units = self._get_valid_units()

    @callback
    def async_update_group_state(self) -> None:
        """Query all members and determine the sensor group state."""
        self.calculate_state_attributes(self._get_valid_entities())
        states: List[str] = []
        valid_units: Set[str] = self._valid_units
        valid_states: List[bool] = []
        sensor_values: List[Tuple[str, float, State]] = []
        for entity_id in self._entity_ids:
            if (state := self.hass.states.get(entity_id)) is not None:
                states.append(state.state)
                try:
                    numeric_state: float = float(state.state)
                    uom: Optional[str] = state.attributes.get('unit_of_measurement')
                    if valid_units and uom in valid_units and (self._can_convert is True):
                        numeric_state = UNIT_CONVERTERS[self.device_class].convert(numeric_state, uom, self.native_unit_of_measurement)
                    if valid_units and uom not in valid_units:
                        raise HomeAssistantError('Not a valid unit')
                    sensor_values.append((entity_id, numeric_state, state))
                    if entity_id in self._state_incorrect:
                        self._state_incorrect.remove(entity_id)
                    valid_states.append(True)
                except ValueError:
                    valid_states.append(False)
                    if not self._ignore_non_numeric and entity_id not in self._state_incorrect:
                        self._state_incorrect.add(entity_id)
                        _LOGGER.warning('Unable to use state. Only numerical states are supported, entity %s with value %s excluded from calculation in %s', entity_id, state.state, self.entity_id)
                    continue
                except (KeyError, HomeAssistantError):
                    valid_states.append(False)
                    if entity_id not in self._state_incorrect:
                        self._state_incorrect.add(entity_id)
                        _LOGGER.warning('Unable to use state. Only entities with correct unit of measurement is supported, entity %s, value %s with device class %s and unit of measurement %s excluded from calculation in %s', entity_id, state.state, self.device_class, state.attributes.get('unit_of_measurement'), self.entity_id)
        self._attr_available = any((numeric_state for numeric_state in valid_states))
        valid_state: bool = self.mode((state not in (STATE_UNKNOWN, STATE_UNAVAILABLE) for state in states))
        valid_state_numeric: bool = self.mode((numeric_state for numeric_state in valid_states))
        if not valid_state or not valid_state_numeric:
            self._attr_native_value = None
            return
        self._extra_state_attribute, self._attr_native_value = self._state_calc(sensor_values)

    @property
    def extra_state_attributes(self) -> Dict[str, str]:
        """Return the state attributes of the sensor."""
        return {ATTR_ENTITY_ID: self._entity_ids, **self._extra_state_attribute}

    @property
    def icon(self) -> Optional[str]:
        """Return the icon.

        Only override the icon if the device class is not set.
        """
        if not self.device_class:
            return 'mdi:calculator'
        return None

    def _calculate_state_class(
        self,
        state_class: Optional[str],
        valid_state_entities: List[str],
    ) -> Optional[str]:
        """Calculate state class.

        If user has configured a state class we will use that.
        If a state class is not set then test if same state class
        on source entities and use that.
        Otherwise return no state class.
        """
        if state_class:
            return state_class
        if not valid_state_entities:
            return None
        if not self._ignore_non_numeric and len(valid_state_entities) < len(self._entity_ids):
            return None
        state_classes: List[str] = []
        for entity_id in valid_state_entities:
            try:
                _state_class = get_capability(self.hass, entity_id, 'state_class')
            except HomeAssistantError:
                return None
            if not _state_class:
                return None
            state_classes.append(_state_class)
        if all((x == state_classes[0] for x in state_classes)):
            async_delete_issue(self.hass, SENSOR_DOMAIN, f'{self.entity_id}_state_classes_not_matching')
            return state_classes[0]
        async_create_issue(self.hass, GROUP_DOMAIN, f'{self.entity_id}_state_classes_not_matching', is_fixable=False, is_persistent=False, severity=IssueSeverity.WARNING, translation_key='state_classes_not_matching', translation_placeholders={'entity_id': self.entity_id, 'source_entities': ', '.join(self._entity_ids), 'state_classes': ', '.join(state_classes)})
        return None

    def _calculate_device_class(
        self,
        device_class: Optional[str],
        valid_state_entities: List[str],
    ) -> Optional[str]:
        """Calculate device class.

        If user has configured a device class we will use that.
        If a device class is not set then test if same device class
        on source entities and use that.
        Otherwise return no device class.
        """
        if device_class:
            return device_class
        if not valid_state_entities:
            return None
        if not self._ignore_non_numeric and len(valid_state_entities) < len(self._entity_ids):
            return None
        device_classes: List[str] = []
        for entity_id in valid_state_entities:
            try:
                _device_class = get_device_class(self.hass, entity_id)
            except HomeAssistantError:
                return None
            if not _device_class:
                return None
            device_classes.append(SensorDeviceClass(_device_class))
        if all((x == device_classes[0] for x in device_classes)):
            async_delete_issue(self.hass, SENSOR_DOMAIN, f'{self.entity_id}_device_classes_not_matching')
            return device_classes[0]
        async_create_issue(self.hass, GROUP_DOMAIN, f'{self.entity_id}_device_classes_not_matching', is_fixable=False, is_persistent=False, severity=IssueSeverity.WARNING, translation_key='device_classes_not_matching', translation_placeholders={'entity_id': self.entity_id, 'source_entities': ', '.join(self._entity_ids), 'device_classes': ', '.join(device_classes)})
        return None

    def _calculate_unit_of_measurement(
        self,
        unit_of_measurement: Optional[str],
        valid_state_entities: List[str],
    ) -> Optional[str]:
        """Calculate the unit of measurement.

        If user has configured a unit of measurement we will use that.
        If a device class is set then test if unit of measurements are compatible.
        If no device class or uom's not compatible we will use no unit of measurement.
        """
        if unit_of_measurement:
            return unit_of_measurement
        if not valid_state_entities:
            return None
        if not self._ignore_non_numeric and len(valid_state_entities) < len(self._entity_ids):
            return None
        unit_of_measurements: List[str] = []
        for entity_id in valid_state_entities:
            try:
                _unit_of_measurement = get_unit_of_measurement(self.hass, entity_id)
            except HomeAssistantError:
                return None
            if not _unit_of_measurement:
                return None
            unit_of_measurements.append(_unit_of_measurement)
        if (device_class := self.device_class) in UNIT_CONVERTERS and all((uom in UNIT_CONVERTERS[device_class].VALID_UNITS for uom in unit_of_measurements)) or (device_class and device_class not in UNIT_CONVERTERS and (device_class in DEVICE_CLASS_UNITS) and all((uom in DEVICE_CLASS_UNITS[device_class] for uom in unit_of_measurements))) or (device_class is None and all((x == unit_of_measurements[0] for x in unit_of_measurements))):
            async_delete_issue(self.hass, SENSOR_DOMAIN, f'{self.entity_id}_uoms_not_matching_device_class')
            async_delete_issue(self.hass, SENSOR_DOMAIN, f'{self.entity_id}_uoms_not_matching_no_device_class')
            return unit_of_measurements[0]
        if device_class:
            async_create_issue(self.hass, GROUP_DOMAIN, f'{self.entity_id}_uoms_not_matching_device_class', is_fixable=False, is_persistent=False, severity=IssueSeverity.WARNING, translation_key='uoms_not_matching_device_class', translation_placeholders={'entity_id': self.entity_id, 'device_class': device_class, 'source_entities': ', '.join(self._entity_ids), 'uoms': ', '.join(unit_of_measurements)})
        else:
            async_create_issue(self.hass, GROUP_DOMAIN, f'{self.entity_id}_uoms_not_matching_no_device_class', is_fixable=False, is_persistent=False, severity=IssueSeverity.WARNING, translation_key='uoms_not_matching_no_device_class', translation_placeholders={'entity_id': self.entity_id, 'source_entities': ', '.join(self._entity_ids), 'uoms': ', '.join(unit_of_measurements)})
        return None

    def _get_valid_units(self) -> Set[str]:
        """Return valid units.

        If device class is set and compatible unit of measurements.
        If device class is not set, use one unit of measurement.
        Only calculate valid units if there are no valid units set.
        """
        if (valid_units := self._valid_units) and (not self._ignore_non_numeric):
            return valid_units
        native_uom: Optional[str] = self.native_unit_of_measurement
        if (device_class := self.device_class) in UNIT_CONVERTERS and native_uom:
            self._can_convert = True
            return UNIT_CONVERTERS[device_class].VALID_UNITS
        if device_class and device_class in DEVICE_CLASS_UNITS and native_uom:
            valid_uoms: Set[str] = DEVICE_CLASS_UNITS[device_class]
            return valid_uoms
        if device_class is None and native_uom:
            return {native_uom}
        return set()

    def _get_valid_entities(self) -> List[str]:
        """Return list of valid entities."""
        return [entity_id for entity_id in self._entity_ids if _has_numeric_state(self.hass, entity_id)]
