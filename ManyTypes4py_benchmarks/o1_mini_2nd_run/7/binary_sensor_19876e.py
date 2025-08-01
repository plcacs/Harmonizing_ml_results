"""Support for monitoring if a sensor value is below/above a threshold."""
from __future__ import annotations
from collections.abc import Callable, Mapping
import logging
from typing import Any, Final, Optional, Union, Dict
import voluptuous as vol
from homeassistant.components.binary_sensor import (
    DEVICE_CLASSES_SCHEMA,
    PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA,
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CONF_DEVICE_CLASS,
    CONF_ENTITY_ID,
    CONF_NAME,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    callback,
)
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
)
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
    AddEntitiesCallback,
)
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.typing import (
    ConfigType,
    DiscoveryInfoType,
)
from .const import (
    ATTR_HYSTERESIS,
    ATTR_LOWER,
    ATTR_POSITION,
    ATTR_SENSOR_VALUE,
    ATTR_TYPE,
    ATTR_UPPER,
    CONF_HYSTERESIS,
    CONF_LOWER,
    CONF_UPPER,
    DEFAULT_HYSTERESIS,
    POSITION_ABOVE,
    POSITION_BELOW,
    POSITION_IN_RANGE,
    POSITION_UNKNOWN,
    TYPE_LOWER,
    TYPE_RANGE,
    TYPE_UPPER,
)
_LOGGER = logging.getLogger(__name__)
DEFAULT_NAME: Final[str] = 'Threshold'

def no_missing_threshold(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate data point list is greater than polynomial degrees."""
    if value.get(CONF_LOWER) is None and value.get(CONF_UPPER) is None:
        raise vol.Invalid('Lower or Upper thresholds are not provided')
    return value

PLATFORM_SCHEMA = vol.All(
    BINARY_SENSOR_PLATFORM_SCHEMA.extend({
        vol.Required(CONF_ENTITY_ID): cv.entity_id,
        vol.Optional(CONF_DEVICE_CLASS): DEVICE_CLASSES_SCHEMA,
        vol.Optional(CONF_HYSTERESIS, default=DEFAULT_HYSTERESIS): vol.Coerce(float),
        vol.Optional(CONF_LOWER): vol.Coerce(float),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_UPPER): vol.Coerce(float),
    }),
    no_missing_threshold,
)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Initialize threshold config entry."""
    registry = er.async_get(hass)
    device_class: Optional[BinarySensorDeviceClass] = None
    entity_id: str = er.async_validate_entity_id(registry, config_entry.options[CONF_ENTITY_ID])
    device_info: DeviceInfo = async_device_info_to_link_from_entity(hass, entity_id)
    hysteresis: float = config_entry.options[CONF_HYSTERESIS]
    lower: Optional[float] = config_entry.options.get(CONF_LOWER)
    name: str = config_entry.title
    unique_id: str = config_entry.entry_id
    upper: Optional[float] = config_entry.options.get(CONF_UPPER)
    async_add_entities([
        ThresholdSensor(
            entity_id=entity_id,
            name=name,
            lower=lower,
            upper=upper,
            hysteresis=hysteresis,
            device_class=device_class,
            unique_id=unique_id,
            device_info=device_info,
        )
    ])

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Threshold sensor."""
    entity_id: str = config[CONF_ENTITY_ID]
    name: str = config[CONF_NAME]
    lower: Optional[float] = config.get(CONF_LOWER)
    upper: Optional[float] = config.get(CONF_UPPER)
    hysteresis: float = config[CONF_HYSTERESIS]
    device_class: Optional[BinarySensorDeviceClass] = config.get(CONF_DEVICE_CLASS)
    async_add_entities([
        ThresholdSensor(
            entity_id=entity_id,
            name=name,
            lower=lower,
            upper=upper,
            hysteresis=hysteresis,
            device_class=device_class,
            unique_id=None,
        )
    ])

def _threshold_type(lower: Optional[float], upper: Optional[float]) -> str:
    """Return the type of threshold this sensor represents."""
    if lower is not None and upper is not None:
        return TYPE_RANGE
    if lower is not None:
        return TYPE_LOWER
    return TYPE_UPPER

class ThresholdSensor(BinarySensorEntity):
    """Representation of a Threshold sensor."""
    _attr_should_poll: bool = False
    _unrecorded_attributes: frozenset = frozenset({
        ATTR_ENTITY_ID,
        ATTR_HYSTERESIS,
        ATTR_LOWER,
        ATTR_TYPE,
        ATTR_UPPER,
    })

    def __init__(
        self,
        entity_id: str,
        name: str,
        lower: Optional[float],
        upper: Optional[float],
        hysteresis: float,
        device_class: Optional[BinarySensorDeviceClass],
        unique_id: Optional[str],
        device_info: Optional[DeviceInfo] = None,
    ) -> None:
        """Initialize the Threshold sensor."""
        self._preview_callback: Optional[Callable[[Union[str, bool], Dict[str, Any]], None]] = None
        self._attr_unique_id: Optional[str] = unique_id
        self._attr_device_info: Optional[DeviceInfo] = device_info
        self._entity_id: str = entity_id
        self._attr_name: str = name
        if lower is not None:
            self._threshold_lower: float = lower
        if upper is not None:
            self._threshold_upper: float = upper
        self.threshold_type: str = _threshold_type(lower, upper)
        self._hysteresis: float = hysteresis
        self._attr_device_class: Optional[BinarySensorDeviceClass] = device_class
        self._state_position: str = POSITION_UNKNOWN
        self.sensor_value: Optional[float] = None

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        self._async_setup_sensor()

    @callback
    def _async_setup_sensor(self) -> None:
        """Set up the sensor and start tracking state changes."""

        def _update_sensor_state() -> None:
            """Handle sensor state changes."""
            new_state = self.hass.states.get(self._entity_id)
            if new_state is None:
                return
            try:
                self.sensor_value = None if new_state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE] else float(new_state.state)
            except (ValueError, TypeError):
                self.sensor_value = None
                _LOGGER.warning('State is not numerical')
            self._update_state()
            if self._preview_callback:
                calculated_state = self._async_calculate_state()
                self._preview_callback(calculated_state.state, calculated_state.attributes)

        @callback
        def async_threshold_sensor_state_listener(event: Event) -> None:
            """Handle sensor state changes."""
            _update_sensor_state()
            if not self._preview_callback:
                self.async_write_ha_state()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                [self._entity_id],
                async_threshold_sensor_state_listener,
            )
        )
        _update_sensor_state()

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the sensor."""
        return {
            ATTR_ENTITY_ID: self._entity_id,
            ATTR_HYSTERESIS: self._hysteresis,
            ATTR_LOWER: getattr(self, '_threshold_lower', None),
            ATTR_POSITION: self._state_position,
            ATTR_SENSOR_VALUE: self.sensor_value,
            ATTR_TYPE: self.threshold_type,
            ATTR_UPPER: getattr(self, '_threshold_upper', None),
        }

    @callback
    def _update_state(self) -> None:
        """Update the state."""

        def below(sensor_value: float, threshold: float) -> bool:
            """Determine if the sensor value is below a threshold."""
            return sensor_value < threshold - self._hysteresis

        def above(sensor_value: float, threshold: float) -> bool:
            """Determine if the sensor value is above a threshold."""
            return sensor_value > threshold + self._hysteresis

        if self.sensor_value is None:
            self._state_position = POSITION_UNKNOWN
            self._attr_is_on = None
            return

        if self.threshold_type == TYPE_LOWER:
            if self._attr_is_on is None:
                self._attr_is_on = False
                self._state_position = POSITION_ABOVE
            if below(self.sensor_value, self._threshold_lower):
                self._state_position = POSITION_BELOW
                self._attr_is_on = True
            elif above(self.sensor_value, self._threshold_lower):
                self._state_position = POSITION_ABOVE
                self._attr_is_on = False
            return

        if self.threshold_type == TYPE_UPPER:
            assert hasattr(self, '_threshold_upper') and self._threshold_upper is not None
            if self._attr_is_on is None:
                self._attr_is_on = False
                self._state_position = POSITION_BELOW
            if above(self.sensor_value, self._threshold_upper):
                self._state_position = POSITION_ABOVE
                self._attr_is_on = True
            elif below(self.sensor_value, self._threshold_upper):
                self._state_position = POSITION_BELOW
                self._attr_is_on = False
            return

        if self.threshold_type == TYPE_RANGE:
            if self._attr_is_on is None:
                self._attr_is_on = True
                self._state_position = POSITION_IN_RANGE
            if below(self.sensor_value, self._threshold_lower):
                self._state_position = POSITION_BELOW
                self._attr_is_on = False
            if above(self.sensor_value, self._threshold_upper):
                self._state_position = POSITION_ABOVE
                self._attr_is_on = False
            elif (
                above(self.sensor_value, self._threshold_lower)
                and below(self.sensor_value, self._threshold_upper)
            ):
                self._state_position = POSITION_IN_RANGE
                self._attr_is_on = True
            return

    @callback
    def async_start_preview(
        self,
        preview_callback: Callable[[Union[str, bool], Dict[str, Any]], None],
    ) -> CALLBACK_TYPE:
        """Render a preview."""
        if (
            not self._entity_id
            or (not hasattr(self, '_threshold_lower') and not hasattr(self, '_threshold_upper'))
        ):
            self._attr_available = False
            calculated_state = self._async_calculate_state()
            preview_callback(calculated_state.state, calculated_state.attributes)
            return self._call_on_remove_callbacks
        self._preview_callback = preview_callback
        self._async_setup_sensor()
        return self._call_on_remove_callbacks

    def _async_calculate_state(self) -> BinarySensorEntity:
        """Calculate the state for preview."""
        # Implementation of _async_calculate_state is missing in the original code.
        # Adding a stub to satisfy type hints.
        return self
