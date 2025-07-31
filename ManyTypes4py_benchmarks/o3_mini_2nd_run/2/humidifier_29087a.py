"""Adds support for generic hygrostat units."""
from __future__ import annotations
import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
import logging
from typing import Any, Optional, cast
from homeassistant.components.humidifier import (
    ATTR_HUMIDITY,
    MODE_AWAY,
    MODE_NORMAL,
    PLATFORM_SCHEMA as HUMIDIFIER_PLATFORM_SCHEMA,
    HumidifierAction,
    HumidifierDeviceClass,
    HumidifierEntity,
    HumidifierEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_MODE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import (
    DOMAIN as HOMEASSISTANT_DOMAIN,
    Event,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers import condition, config_validation as cv
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_state_report_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import (
    CONF_AWAY_FIXED,
    CONF_AWAY_HUMIDITY,
    CONF_DEVICE_CLASS,
    CONF_DRY_TOLERANCE,
    CONF_HUMIDIFIER,
    CONF_INITIAL_STATE,
    CONF_KEEP_ALIVE,
    CONF_MAX_HUMIDITY,
    CONF_MIN_DUR,
    CONF_MIN_HUMIDITY,
    CONF_SENSOR,
    CONF_STALE_DURATION,
    CONF_TARGET_HUMIDITY,
    CONF_WET_TOLERANCE,
    HYGROSTAT_SCHEMA,
)
_LOGGER = logging.getLogger(__name__)
ATTR_SAVED_HUMIDITY = 'saved_humidity'
PLATFORM_SCHEMA = HUMIDIFIER_PLATFORM_SCHEMA.extend(HYGROSTAT_SCHEMA.schema)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the generic hygrostat platform."""
    if discovery_info:
        config = discovery_info
    await _async_setup_config(hass, config, config.get(CONF_UNIQUE_ID), async_add_entities)


async def async_setup_entry(
    hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Initialize config entry."""
    await _async_setup_config(hass, config_entry.options, config_entry.entry_id, async_add_entities)


def _time_period_or_none(value: Any) -> Optional[timedelta]:
    if value is None:
        return None
    return cast(timedelta, cv.time_period(value))


async def _async_setup_config(
    hass: HomeAssistant, config: ConfigType, unique_id: str, async_add_entities: AddEntitiesCallback
) -> None:
    name: str = config[CONF_NAME]
    switch_entity_id: str = config[CONF_HUMIDIFIER]
    sensor_entity_id: str = config[CONF_SENSOR]
    min_humidity: Optional[float] = config.get(CONF_MIN_HUMIDITY)
    max_humidity: Optional[float] = config.get(CONF_MAX_HUMIDITY)
    target_humidity: Optional[float] = config.get(CONF_TARGET_HUMIDITY)
    device_class: Optional[HumidifierDeviceClass] = config.get(CONF_DEVICE_CLASS)
    min_cycle_duration: Optional[timedelta] = _time_period_or_none(config.get(CONF_MIN_DUR))
    sensor_stale_duration: Optional[timedelta] = _time_period_or_none(config.get(CONF_STALE_DURATION))
    dry_tolerance: float = config[CONF_DRY_TOLERANCE]
    wet_tolerance: float = config[CONF_WET_TOLERANCE]
    keep_alive: Optional[timedelta] = _time_period_or_none(config.get(CONF_KEEP_ALIVE))
    initial_state: Optional[bool] = config.get(CONF_INITIAL_STATE)
    away_humidity: Optional[float] = config.get(CONF_AWAY_HUMIDITY)
    away_fixed: bool = config.get(CONF_AWAY_FIXED)
    async_add_entities(
        [
            GenericHygrostat(
                hass,
                name,
                switch_entity_id,
                sensor_entity_id,
                min_humidity,
                max_humidity,
                target_humidity,
                device_class,
                min_cycle_duration,
                dry_tolerance,
                wet_tolerance,
                keep_alive,
                initial_state,
                away_humidity,
                away_fixed,
                sensor_stale_duration,
                unique_id,
            )
        ]
    )


class GenericHygrostat(HumidifierEntity, RestoreEntity):
    """Representation of a Generic Hygrostat device."""

    _attr_should_poll: bool = False

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        switch_entity_id: str,
        sensor_entity_id: str,
        min_humidity: Optional[float],
        max_humidity: Optional[float],
        target_humidity: Optional[float],
        device_class: Optional[HumidifierDeviceClass],
        min_cycle_duration: Optional[timedelta],
        dry_tolerance: float,
        wet_tolerance: float,
        keep_alive: Optional[timedelta],
        initial_state: Optional[bool],
        away_humidity: Optional[float],
        away_fixed: bool,
        sensor_stale_duration: Optional[timedelta],
        unique_id: str,
    ) -> None:
        """Initialize the hygrostat."""
        self._name: str = name
        self._switch_entity_id: str = switch_entity_id
        self._sensor_entity_id: str = sensor_entity_id
        self._attr_device_info = async_device_info_to_link_from_entity(hass, switch_entity_id)
        self._device_class: HumidifierDeviceClass = device_class or HumidifierDeviceClass.HUMIDIFIER
        self._min_cycle_duration: Optional[timedelta] = min_cycle_duration
        self._dry_tolerance: float = dry_tolerance
        self._wet_tolerance: float = wet_tolerance
        self._keep_alive: Optional[timedelta] = keep_alive
        self._state: Optional[bool] = initial_state
        self._saved_target_humidity: Optional[float] = away_humidity or target_humidity
        self._active: bool = False
        self._cur_humidity: Optional[float] = None
        self._humidity_lock: asyncio.Lock = asyncio.Lock()
        self._min_humidity: Optional[float] = min_humidity
        self._max_humidity: Optional[float] = max_humidity
        self._target_humidity: Optional[float] = target_humidity
        if away_humidity:
            self._attr_supported_features |= HumidifierEntityFeature.MODES
        self._away_humidity: Optional[float] = away_humidity
        self._away_fixed: bool = away_fixed
        self._sensor_stale_duration: Optional[timedelta] = sensor_stale_duration
        self._remove_stale_tracking: Optional[Callable[[], None]] = None
        self._is_away: bool = False
        self._attr_action: HumidifierAction = HumidifierAction.IDLE
        self._attr_unique_id: str = unique_id

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_track_state_change_event(self.hass, self._sensor_entity_id, self._async_sensor_event)
        )
        self.async_on_remove(
            async_track_state_report_event(self.hass, self._sensor_entity_id, self._async_sensor_event)
        )
        self.async_on_remove(
            async_track_state_change_event(self.hass, self._switch_entity_id, self._async_switch_event)
        )
        if self._keep_alive:
            self.async_on_remove(
                async_track_time_interval(self.hass, self._async_operate, self._keep_alive)
            )

        async def _async_startup(event: Optional[Event]) -> None:
            """Init on startup."""
            sensor_state: Optional[State] = self.hass.states.get(self._sensor_entity_id)
            if sensor_state is None or sensor_state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                _LOGGER.debug(
                    'The sensor state is %s, initialization is delayed',
                    sensor_state.state if sensor_state is not None else 'None',
                )
                return
            await self._async_sensor_update(sensor_state)

        self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)
        old_state: Optional[State] = await self.async_get_last_state()
        if old_state is not None:
            if old_state.attributes.get(ATTR_MODE) == MODE_AWAY:
                self._is_away = True
                self._saved_target_humidity = self._target_humidity
                self._target_humidity = self._away_humidity or self._target_humidity
            if old_state.attributes.get(ATTR_HUMIDITY):
                self._target_humidity = int(old_state.attributes[ATTR_HUMIDITY])
            if old_state.attributes.get(ATTR_SAVED_HUMIDITY):
                self._saved_target_humidity = int(old_state.attributes[ATTR_SAVED_HUMIDITY])
            if old_state.state:
                self._state = old_state.state == STATE_ON
        if self._target_humidity is None:
            if self._device_class == HumidifierDeviceClass.HUMIDIFIER:
                self._target_humidity = self.min_humidity
            else:
                self._target_humidity = self.max_humidity
            _LOGGER.warning('No previously saved humidity, setting to %s', self._target_humidity)
        if self._state is None:
            self._state = False
        await _async_startup(None)

    async def async_will_remove_from_hass(self) -> Any:
        """Run when entity will be removed from hass."""
        if self._remove_stale_tracking:
            self._remove_stale_tracking()
        return await super().async_will_remove_from_hass()

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._active

    @property
    def extra_state_attributes(self) -> Optional[Mapping[str, Any]]:
        """Return the optional state attributes."""
        if self._saved_target_humidity is not None:
            return {ATTR_SAVED_HUMIDITY: self._saved_target_humidity}
        return None

    @property
    def name(self) -> str:
        """Return the name of the hygrostat."""
        return self._name

    @property
    def is_on(self) -> bool:
        """Return true if the hygrostat is on."""
        return bool(self._state)

    @property
    def current_humidity(self) -> Optional[float]:
        """Return the measured humidity."""
        return self._cur_humidity

    @property
    def target_humidity(self) -> Optional[float]:
        """Return the humidity we try to reach."""
        return self._target_humidity

    @property
    def mode(self) -> Optional[str]:
        """Return the current mode."""
        if self._away_humidity is None:
            return None
        if self._is_away:
            return MODE_AWAY
        return MODE_NORMAL

    @property
    def available_modes(self) -> Optional[list[str]]:
        """Return a list of available modes."""
        if self._away_humidity:
            return [MODE_NORMAL, MODE_AWAY]
        return None

    @property
    def device_class(self) -> HumidifierDeviceClass:
        """Return the device class of the humidifier."""
        return self._device_class

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn hygrostat on."""
        if not self._active:
            return
        self._state = True
        await self._async_operate(force=True)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn hygrostat off."""
        if not self._active:
            return
        self._state = False
        if self._is_device_active:
            await self._async_device_turn_off()
        self.async_write_ha_state()

    async def async_set_humidity(self, humidity: Optional[float]) -> None:
        """Set new target humidity."""
        if humidity is None:
            return
        if self._is_away and self._away_fixed:
            self._saved_target_humidity = humidity
            self.async_write_ha_state()
            return
        self._target_humidity = humidity
        await self._async_operate()
        self.async_write_ha_state()

    @property
    def min_humidity(self) -> Optional[float]:
        """Return the minimum humidity."""
        if self._min_humidity is not None:
            return self._min_humidity
        return super().min_humidity

    @property
    def max_humidity(self) -> Optional[float]:
        """Return the maximum humidity."""
        if self._max_humidity is not None:
            return self._max_humidity
        return super().max_humidity

    async def _async_sensor_event(self, event: Event) -> None:
        """Handle ambient humidity changes."""
        new_state = event.data['new_state']
        if new_state is None:
            return
        await self._async_sensor_update(new_state)

    async def _async_sensor_update(self, new_state: State) -> None:
        """Update state based on humidity sensor."""
        if self._sensor_stale_duration:
            if self._remove_stale_tracking:
                self._remove_stale_tracking()
            self._remove_stale_tracking = async_track_time_interval(
                self.hass, self._async_sensor_not_responding, self._sensor_stale_duration
            )
        await self._async_update_humidity(new_state.state)
        await self._async_operate()
        self.async_write_ha_state()

    async def _async_sensor_not_responding(self, now: Optional[datetime] = None) -> None:
        """Handle sensor stale event."""
        state: Optional[State] = self.hass.states.get(self._sensor_entity_id)
        _LOGGER.debug('Sensor has not been updated for %s', now - state.last_reported if now and state else '---')
        _LOGGER.warning('Sensor is stalled, call the emergency stop')
        await self._async_update_humidity('Stalled')

    @callback
    def _async_switch_event(self, event: Event) -> None:
        """Handle humidifier switch state changes."""
        self._async_switch_changed(event.data['new_state'])

    @callback
    def _async_switch_changed(self, new_state: Optional[State]) -> None:
        """Handle humidifier switch state changes."""
        if new_state is None:
            return
        if new_state.state == STATE_ON:
            if self._device_class == HumidifierDeviceClass.DEHUMIDIFIER:
                self._attr_action = HumidifierAction.DRYING
            else:
                self._attr_action = HumidifierAction.HUMIDIFYING
        else:
            self._attr_action = HumidifierAction.IDLE
        self.async_write_ha_state()

    async def _async_update_humidity(self, humidity: str) -> None:
        """Update hygrostat with latest state from sensor."""
        try:
            self._cur_humidity = float(humidity)
        except ValueError as ex:
            if self._active:
                _LOGGER.warning('Unable to update from sensor: %s', ex)
                self._active = False
            else:
                _LOGGER.debug('Unable to update from sensor: %s', ex)
            self._cur_humidity = None
            if self._is_device_active:
                await self._async_device_turn_off()

    async def _async_operate(self, time: Optional[datetime] = None, force: bool = False) -> None:
        """Check if we need to turn humidifying on or off."""
        async with self._humidity_lock:
            if not self._active and None not in (self._cur_humidity, self._target_humidity):
                self._active = True
                force = True
                _LOGGER.debug(
                    'Obtained current and target humidity. Generic hygrostat active. %s, %s',
                    self._cur_humidity,
                    self._target_humidity,
                )
            if not self._active or not self._state:
                return
            if not force and time is None:
                if self._min_cycle_duration:
                    current_state: str = STATE_ON if self._is_device_active else STATE_OFF
                    long_enough: bool = condition.state(self.hass, self._switch_entity_id, current_state, self._min_cycle_duration)
                    if not long_enough:
                        return
            if force:
                dry_tolerance: float = 0
                wet_tolerance: float = 0
            else:
                dry_tolerance = self._dry_tolerance
                wet_tolerance = self._wet_tolerance
            if self._target_humidity is None or self._cur_humidity is None:
                return
            too_dry: bool = self._target_humidity - self._cur_humidity >= dry_tolerance
            too_wet: bool = self._cur_humidity - self._target_humidity >= wet_tolerance
            if self._is_device_active:
                if (self._device_class == HumidifierDeviceClass.HUMIDIFIER and too_wet) or (
                    self._device_class == HumidifierDeviceClass.DEHUMIDIFIER and too_dry
                ):
                    _LOGGER.debug('Turning off humidifier %s', self._switch_entity_id)
                    await self._async_device_turn_off()
                elif time is not None:
                    await self._async_device_turn_on()
            elif (self._device_class == HumidifierDeviceClass.HUMIDIFIER and too_dry) or (
                self._device_class == HumidifierDeviceClass.DEHUMIDIFIER and too_wet
            ):
                _LOGGER.debug('Turning on humidifier %s', self._switch_entity_id)
                await self._async_device_turn_on()
            elif time is not None:
                await self._async_device_turn_off()

    @property
    def _is_device_active(self) -> bool:
        """If the toggleable device is currently active."""
        return self.hass.states.is_state(self._switch_entity_id, STATE_ON)

    async def _async_device_turn_on(self) -> None:
        """Turn humidifier toggleable device on."""
        data: dict[str, Any] = {ATTR_ENTITY_ID: self._switch_entity_id}
        await self.hass.services.async_call(HOMEASSISTANT_DOMAIN, SERVICE_TURN_ON, data)

    async def _async_device_turn_off(self) -> None:
        """Turn humidifier toggleable device off."""
        data: dict[str, Any] = {ATTR_ENTITY_ID: self._switch_entity_id}
        await self.hass.services.async_call(HOMEASSISTANT_DOMAIN, SERVICE_TURN_OFF, data)

    async def async_set_mode(self, mode: str) -> None:
        """Set new mode.

        This method must be run in the event loop and returns a coroutine.
        """
        if self._away_humidity is None:
            return
        if mode == MODE_AWAY and (not self._is_away):
            self._is_away = True
            if not self._saved_target_humidity:
                self._saved_target_humidity = self._away_humidity
            self._saved_target_humidity, self._target_humidity = (self._target_humidity, self._saved_target_humidity)
            await self._async_operate(force=True)
        elif mode == MODE_NORMAL and self._is_away:
            self._is_away = False
            self._saved_target_humidity, self._target_humidity = (self._target_humidity, self._saved_target_humidity)
            await self._async_operate(force=True)
        self.async_write_ha_state()