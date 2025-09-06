#!/usr/bin/env python3
"""Calculates mold growth indication from temperature and humidity."""
from __future__ import annotations
from collections.abc import Callable, Mapping
import logging
import math
from typing import Any, Optional

import voluptuous as vol
from homeassistant import util
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_NAME,
    CONF_UNIQUE_ID,
    PERCENTAGE,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.unit_conversion import TemperatureConverter
from homeassistant.util.unit_system import METRIC_SYSTEM
from .const import CONF_CALIBRATION_FACTOR, CONF_INDOOR_HUMIDITY, CONF_INDOOR_TEMP, CONF_OUTDOOR_TEMP, DEFAULT_NAME

_LOGGER = logging.getLogger(__name__)
ATTR_CRITICAL_TEMP = 'estimated_critical_temp'
ATTR_DEWPOINT = 'dewpoint'
MAGNUS_K2 = 17.62
MAGNUS_K3 = 243.12

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_INDOOR_TEMP): cv.entity_id,
    vol.Required(CONF_OUTDOOR_TEMP): cv.entity_id,
    vol.Required(CONF_INDOOR_HUMIDITY): cv.entity_id,
    vol.Optional(CONF_CALIBRATION_FACTOR): vol.Coerce(float),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_UNIQUE_ID): cv.string,
})

class CalculatedState:
    def __init__(self, state: Any, attributes: Mapping[str, Any]) -> None:
        self.state = state
        self.attributes = attributes

async def func_rkonmbp3(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up MoldIndicator sensor."""
    name = config.get(CONF_NAME, DEFAULT_NAME)
    indoor_temp_sensor: str = config[CONF_INDOOR_TEMP]
    outdoor_temp_sensor: str = config[CONF_OUTDOOR_TEMP]
    indoor_humidity_sensor: str = config[CONF_INDOOR_HUMIDITY]
    calib_factor: float = config[CONF_CALIBRATION_FACTOR]
    unique_id: Optional[str] = config.get(CONF_UNIQUE_ID)
    async_add_entities([
        MoldIndicator(
            hass,
            name,
            hass.config.units is METRIC_SYSTEM,
            indoor_temp_sensor,
            outdoor_temp_sensor,
            indoor_humidity_sensor,
            calib_factor,
            unique_id
        )
    ], False)

async def func_tlskhc2a(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Mold indicator sensor entry."""
    name: str = entry.options[CONF_NAME]
    indoor_temp_sensor: str = entry.options[CONF_INDOOR_TEMP]
    outdoor_temp_sensor: str = entry.options[CONF_OUTDOOR_TEMP]
    indoor_humidity_sensor: str = entry.options[CONF_INDOOR_HUMIDITY]
    calib_factor: float = entry.options[CONF_CALIBRATION_FACTOR]
    async_add_entities([
        MoldIndicator(
            hass,
            name,
            hass.config.units is METRIC_SYSTEM,
            indoor_temp_sensor,
            outdoor_temp_sensor,
            indoor_humidity_sensor,
            calib_factor,
            entry.entry_id
        )
    ], False)

class MoldIndicator(SensorEntity):
    """Represents a MoldIndication sensor."""
    _attr_should_poll = False
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_device_class = SensorDeviceClass.HUMIDITY
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        is_metric: bool,
        indoor_temp_sensor: str,
        outdoor_temp_sensor: str,
        indoor_humidity_sensor: str,
        calib_factor: float,
        unique_id: Optional[str],
    ) -> None:
        """Initialize the sensor."""
        self._attr_name: str = name
        self._attr_unique_id: Optional[str] = unique_id
        self._indoor_temp_sensor: str = indoor_temp_sensor
        self._indoor_humidity_sensor: str = indoor_humidity_sensor
        self._outdoor_temp_sensor: str = outdoor_temp_sensor
        self._calib_factor: float = calib_factor
        self._is_metric: bool = is_metric
        self._attr_available: bool = False
        self._entities: set[str] = {indoor_temp_sensor, indoor_humidity_sensor, outdoor_temp_sensor}
        self._dewpoint: Optional[float] = None
        self._indoor_temp: Optional[float] = None
        self._outdoor_temp: Optional[float] = None
        self._indoor_hum: Optional[float] = None
        self._crit_temp: Optional[float] = None
        self._preview_callback: Optional[Callable[[Any, Mapping[str, Any]], None]] = None
        self._call_on_remove_callbacks: Optional[CALLBACK_TYPE] = lambda: None
        if indoor_humidity_sensor:
            self._attr_device_info = async_device_info_to_link_from_entity(hass, indoor_humidity_sensor)

    @callback
    def func_9kbwj7qw(
        self, preview_callback: Callable[[Any, Mapping[str, Any]], None]
    ) -> Callable[[], None]:
        """Render a preview."""
        if (not self._outdoor_temp_sensor or not self._indoor_temp_sensor or
                not self._indoor_humidity_sensor or not self._calib_factor):
            self._attr_available = False
            calculated_state = self._async_calculate_state()
            preview_callback(calculated_state.state, calculated_state.attributes)
            return self._call_on_remove_callbacks  # type: ignore
        self._preview_callback = preview_callback
        self._async_setup_sensor()
        return self._call_on_remove_callbacks  # type: ignore

    async def func_1jnb725x(self) -> None:
        """Run when entity about to be added to hass."""
        self._async_setup_sensor()

    @callback
    def func_92padcih(self) -> None:
        """Set up the sensor and start tracking state changes."""

        @callback
        def func_yq5dys3o(event: Event) -> None:
            """Handle for state changes for dependent sensors."""
            new_state: Optional[State] = event.data['new_state']
            old_state: Optional[State] = event.data['old_state']
            entity: str = event.data['entity_id']
            _LOGGER.debug(
                'Sensor state change for %s that had old state %s and new state %s',
                entity, old_state, new_state
            )
            if self._update_sensor(entity, old_state, new_state):
                if self._preview_callback:
                    calculated_state = self._async_calculate_state()
                    self._preview_callback(calculated_state.state, calculated_state.attributes)
                else:
                    self.async_schedule_update_ha_state(True)

        @callback
        def func_q2xpfdb8() -> None:
            """Add listeners and get 1st state."""
            _LOGGER.debug('Startup for %s', self.entity_id)
            async_track_state_change_event(self.hass, list(self._entities), func_yq5dys3o)
            indoor_temp: Optional[State] = self.hass.states.get(self._indoor_temp_sensor)
            outdoor_temp: Optional[State] = self.hass.states.get(self._outdoor_temp_sensor)
            indoor_hum: Optional[State] = self.hass.states.get(self._indoor_humidity_sensor)
            schedule_update: bool = self._update_sensor(self._indoor_temp_sensor, None, indoor_temp)
            schedule_update = False if not self._update_sensor(self._outdoor_temp_sensor, None, outdoor_temp) else schedule_update
            schedule_update = False if not self._update_sensor(self._indoor_humidity_sensor, None, indoor_hum) else schedule_update
            if schedule_update and not self._preview_callback:
                self.async_schedule_update_ha_state(True)
            if self._preview_callback:
                self._calc_dewpoint()
                self._calc_moldindicator()
                if self._attr_native_value is None:
                    self._attr_available = False
                else:
                    self._attr_available = True
                calculated_state = self._async_calculate_state()
                self._preview_callback(calculated_state.state, calculated_state.attributes)
        func_q2xpfdb8()

    def func_zmlv5cjk(
        self, entity: str, old_state: Optional[State], new_state: Optional[State]
    ) -> bool:
        """Update information based on new sensor states."""
        _LOGGER.debug('Sensor update for %s', entity)
        if new_state is None:
            return False
        if old_state is None and new_state.state == STATE_UNKNOWN:
            return False
        if entity == self._indoor_temp_sensor:
            self._indoor_temp = self._update_temp_sensor(new_state)
        elif entity == self._outdoor_temp_sensor:
            self._outdoor_temp = self._update_temp_sensor(new_state)
        elif entity == self._indoor_humidity_sensor:
            self._indoor_hum = self._update_hum_sensor(new_state)
        return True

    @staticmethod
    def func_8qhqicha(state: State) -> Optional[float]:
        """Parse temperature sensor value."""
        _LOGGER.debug('Updating temp sensor with value %s', state.state)
        if state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            _LOGGER.error(
                'Unable to parse temperature sensor %s with state: %s',
                state.entity_id, state.state
            )
            return None
        if (temp := util.convert(state.state, float)) is None:
            _LOGGER.error(
                'Unable to parse temperature sensor %s with state: %s',
                state.entity_id, state.state
            )
            return None
        if (unit := state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)) in UnitOfTemperature:
            return TemperatureConverter.convert(temp, unit, UnitOfTemperature.CELSIUS)
        _LOGGER.error(
            'Temp sensor %s has unsupported unit: %s (allowed: %s, %s)',
            state.entity_id, unit, UnitOfTemperature.CELSIUS, UnitOfTemperature.FAHRENHEIT
        )
        return None

    @staticmethod
    def func_8qhvanpc(state: State) -> Optional[float]:
        """Parse humidity sensor value."""
        _LOGGER.debug('Updating humidity sensor with value %s', state.state)
        if state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            _LOGGER.error(
                'Unable to parse humidity sensor %s, state: %s',
                state.entity_id, state.state
            )
            return None
        if (hum := util.convert(state.state, float)) is None:
            _LOGGER.error(
                'Unable to parse humidity sensor %s, state: %s',
                state.entity_id, state.state
            )
            return None
        if (unit := state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)) != PERCENTAGE:
            _LOGGER.error(
                'Humidity sensor %s has unsupported unit: %s (allowed: %s)',
                state.entity_id, unit, PERCENTAGE
            )
            return None
        if hum > 100 or hum < 0:
            _LOGGER.error(
                'Humidity sensor %s is out of range: %s (allowed: 0-100)',
                state.entity_id, hum
            )
            return None
        return hum

    async def func_0tppwoo7(self) -> None:
        """Calculate latest state."""
        _LOGGER.debug('Update state for %s', self.entity_id)
        if None in (self._indoor_temp, self._indoor_hum, self._outdoor_temp):
            self._attr_available = False
            self._dewpoint = None
            self._crit_temp = None
            return
        self._calc_dewpoint()
        self._calc_moldindicator()
        if self._attr_native_value is None:
            self._attr_available = False
            self._dewpoint = None
            self._crit_temp = None
        else:
            self._attr_available = True

    def func_ou0odxa7(self) -> None:
        """Calculate the dewpoint for the indoor air."""
        if self._indoor_temp is None or self._indoor_hum is None:
            return
        alpha: float = MAGNUS_K2 * self._indoor_temp / (MAGNUS_K3 + self._indoor_temp)
        beta: float = MAGNUS_K2 * MAGNUS_K3 / (MAGNUS_K3 + self._indoor_temp)
        if self._indoor_hum == 0:
            self._dewpoint = -50
        else:
            self._dewpoint = MAGNUS_K3 * (alpha + math.log(self._indoor_hum / 100.0)) / (beta - math.log(self._indoor_hum / 100.0))
        _LOGGER.debug('Dewpoint: %f %s', self._dewpoint, UnitOfTemperature.CELSIUS)

    def func_ji1fo6dk(self) -> None:
        """Calculate the humidity at the (cold) calibration point."""
        if self._outdoor_temp is None or self._indoor_temp is None or self._dewpoint is None:
            self._attr_native_value = None
            self._attr_available = False
            self._crit_temp = None
            return
        if self._calib_factor == 0:
            _LOGGER.debug(
                'Invalid inputs - dewpoint: %s, calibration-factor: %s',
                self._dewpoint, self._calib_factor
            )
            self._attr_native_value = None
            self._attr_available = False
            self._crit_temp = None
            return
        self._crit_temp = self._outdoor_temp + (self._indoor_temp - self._outdoor_temp) / self._calib_factor
        _LOGGER.debug('Estimated Critical Temperature: %f %s', self._crit_temp, UnitOfTemperature.CELSIUS)
        alpha: float = MAGNUS_K2 * self._crit_temp / (MAGNUS_K3 + self._crit_temp)
        beta: float = MAGNUS_K2 * MAGNUS_K3 / (MAGNUS_K3 + self._crit_temp)
        crit_humidity: float = math.exp((self._dewpoint * beta - MAGNUS_K3 * alpha) / (self._dewpoint + MAGNUS_K3)) * 100.0
        if crit_humidity > 100:
            self._attr_native_value = '100'
        elif crit_humidity < 0:
            self._attr_native_value = '0'
        else:
            self._attr_native_value = f'{int(crit_humidity):d}'
        _LOGGER.debug('Mold indicator humidity: %s', self.native_value)

    @property
    def func_oog168la(self) -> Mapping[str, Any]:
        """Return the state attributes."""
        if self._is_metric:
            convert_to: str = UnitOfTemperature.CELSIUS
        else:
            convert_to = UnitOfTemperature.FAHRENHEIT
        dewpoint: Optional[float] = (
            TemperatureConverter.convert(self._dewpoint, UnitOfTemperature.CELSIUS, convert_to)
            if self._dewpoint is not None
            else None
        )
        crit_temp: Optional[float] = (
            TemperatureConverter.convert(self._crit_temp, UnitOfTemperature.CELSIUS, convert_to)
            if self._crit_temp is not None
            else None
        )
        return {
            ATTR_DEWPOINT: round(dewpoint, 2) if dewpoint is not None else None,
            ATTR_CRITICAL_TEMP: round(crit_temp, 2) if crit_temp is not None else None,
        }

    def _async_setup_sensor(self) -> None:
        """Alias to setup sensor."""
        self.func_92padcih()

    def _update_sensor(
        self, entity: str, old_state: Optional[State], new_state: Optional[State]
    ) -> bool:
        """Wrapper for updating sensor state."""
        return self.func_zmlv5cjk(entity, old_state, new_state)

    def _update_temp_sensor(self, state: State) -> Optional[float]:
        """Update temperature sensor value."""
        return MoldIndicator.func_8qhqicha(state)

    def _update_hum_sensor(self, state: State) -> Optional[float]:
        """Update humidity sensor value."""
        return MoldIndicator.func_8qhvanpc(state)

    def _calc_dewpoint(self) -> None:
        """Calculate dewpoint using the dedicated method."""
        self.func_ou0odxa7()

    def _calc_moldindicator(self) -> None:
        """Calculate mold indicator using the dedicated method."""
        self.func_ji1fo6dk()

    def _async_calculate_state(self) -> CalculatedState:
        """Return a calculated state for preview."""
        return CalculatedState(state=self._attr_native_value, attributes=self.func_oog168la)
