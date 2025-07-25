"""Support for Tado hot water zones."""
from __future__ import annotations

import logging
from typing import Any, cast

import voluptuous as vol
from homeassistant.components.water_heater import WaterHeaterEntity, WaterHeaterEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import TadoConfigEntry
from .const import CONST_HVAC_HEAT, CONST_MODE_AUTO, CONST_MODE_HEAT, CONST_MODE_OFF, CONST_MODE_SMART_SCHEDULE, CONST_OVERLAY_MANUAL, CONST_OVERLAY_TADO_MODE, CONST_OVERLAY_TIMER, TYPE_HOT_WATER
from .coordinator import TadoDataUpdateCoordinator
from .entity import TadoZoneEntity
from .helper import decide_duration, decide_overlay_mode
from .repairs import manage_water_heater_fallback_issue

_LOGGER = logging.getLogger(__name__)

MODE_AUTO = "auto"
MODE_HEAT = "heat"
MODE_OFF = "off"
OPERATION_MODES = [MODE_AUTO, MODE_HEAT, MODE_OFF]

WATER_HEATER_MAP_TADO = {
    CONST_OVERLAY_MANUAL: MODE_HEAT,
    CONST_OVERLAY_TIMER: MODE_HEAT,
    CONST_OVERLAY_TADO_MODE: MODE_HEAT,
    CONST_HVAC_HEAT: MODE_HEAT,
    CONST_MODE_SMART_SCHEDULE: MODE_AUTO,
    CONST_MODE_OFF: MODE_OFF,
}

SERVICE_WATER_HEATER_TIMER = "set_water_heater_timer"
ATTR_TIME_PERIOD = "time_period"
WATER_HEATER_TIMER_SCHEMA: dict[vol.Optional | vol.Required, Any] = {
    vol.Required(ATTR_TIME_PERIOD, default="01:00:00"): vol.All(
        cv.time_period, cv.positive_timedelta, lambda td: td.total_seconds()
    ),
    vol.Optional(ATTR_TEMPERATURE): vol.Coerce(float),
}

async def async_setup_entry(
    hass: HomeAssistant,
    entry: TadoConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Tado water heater platform."""
    data = entry.runtime_data
    coordinator = data.coordinator
    entities = await _generate_entities(coordinator)
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_WATER_HEATER_TIMER,
        WATER_HEATER_TIMER_SCHEMA,
        "set_timer",
    )
    async_add_entities(entities, True)
    manage_water_heater_fallback_issue(
        hass=hass,
        water_heater_names=[e.zone_name for e in entities],
        integration_overlay_fallback=coordinator.fallback,
    )

async def _generate_entities(coordinator: TadoDataUpdateCoordinator) -> list[TadoWaterHeater]:
    """Create all water heater entities."""
    entities: list[TadoWaterHeater] = []
    for zone in coordinator.zones:
        if zone["type"] == TYPE_HOT_WATER:
            entity = await create_water_heater_entity(
                coordinator, zone["name"], zone["id"], str(zone["name"])
            )
            entities.append(entity)
    return entities

async def create_water_heater_entity(
    coordinator: TadoDataUpdateCoordinator,
    name: str,
    zone_id: int,
    zone: str,
) -> TadoWaterHeater:
    """Create a Tado water heater device."""
    capabilities = await coordinator.get_capabilities(zone_id)
    supports_temperature_control = capabilities["canSetTemperature"]
    if supports_temperature_control and "temperatures" in capabilities:
        temperatures = capabilities["temperatures"]
        min_temp = float(temperatures["celsius"]["min"])
        max_temp = float(temperatures["celsius"]["max"])
    else:
        min_temp = None
        max_temp = None
    return TadoWaterHeater(
        coordinator, name, zone_id, supports_temperature_control, min_temp, max_temp
    )

class TadoWaterHeater(TadoZoneEntity, WaterHeaterEntity):
    """Representation of a Tado water heater."""

    _attr_name: str | None = None
    _attr_operation_list: list[str] = OPERATION_MODES
    _attr_temperature_unit: str = UnitOfTemperature.CELSIUS

    def __init__(
        self,
        coordinator: TadoDataUpdateCoordinator,
        zone_name: str,
        zone_id: int,
        supports_temperature_control: bool,
        min_temp: float | None,
        max_temp: float | None,
    ) -> None:
        """Initialize of Tado water heater entity."""
        super().__init__(zone_name, coordinator.home_id, zone_id, coordinator)
        self.zone_id = zone_id
        self._attr_unique_id = f"{zone_id} {coordinator.home_id}"
        self._device_is_active = False
        self._supports_temperature_control = supports_temperature_control
        self._min_temperature = min_temp
        self._max_temperature = max_temp
        self._target_temp: float | None = None
        self._attr_supported_features = WaterHeaterEntityFeature.OPERATION_MODE
        if self._supports_temperature_control:
            self._attr_supported_features |= WaterHeaterEntityFeature.TARGET_TEMPERATURE
        self._current_tado_hvac_mode: str = CONST_MODE_SMART_SCHEDULE
        self._overlay_mode: str = CONST_MODE_SMART_SCHEDULE
        self._tado_zone_data: dict[str, Any] | None = None
        self._async_update_data()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self._async_update_data()
        super()._handle_coordinator_update()

    @property
    def current_operation(self) -> str | None:
        """Return current readable operation mode."""
        return WATER_HEATER_MAP_TADO.get(self._current_tado_hvac_mode)

    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._tado_zone_data.target_temp if self._tado_zone_data else None

    @property
    def is_away_mode_on(self) -> bool | None:
        """Return true if away mode is on."""
        return self._tado_zone_data.is_away if self._tado_zone_data else None

    @property
    def min_temp(self) -> float | None:
        """Return the minimum temperature."""
        return self._min_temperature

    @property
    def max_temp(self) -> float | None:
        """Return the maximum temperature."""
        return self._max_temperature

    async def async_set_operation_mode(self, operation_mode: str) -> None:
        """Set new operation mode."""
        mode = None
        if operation_mode == MODE_OFF:
            mode = CONST_MODE_OFF
        elif operation_mode == MODE_AUTO:
            mode = CONST_MODE_SMART_SCHEDULE
        elif operation_mode == MODE_HEAT:
            mode = CONST_MODE_HEAT
        await self._control_heater(hvac_mode=mode)
        await self.coordinator.async_request_refresh()

    async def set_timer(self, time_period: float, temperature: float | None = None) -> None:
        """Set the timer on the entity, and temperature if supported."""
        if not self._supports_temperature_control and temperature is not None:
            temperature = None
        await self._control_heater(
            hvac_mode=CONST_MODE_HEAT,
            target_temp=temperature,
            duration=time_period,
        )
        await self.coordinator.async_request_refresh()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if not self._supports_temperature_control or temperature is None:
            return
        if self._current_tado_hvac_mode not in (
            CONST_MODE_OFF,
            CONST_MODE_AUTO,
            CONST_MODE_SMART_SCHEDULE,
        ):
            await self._control_heater(target_temp=temperature)
            return
        await self._control_heater(
            target_temp=temperature, hvac_mode=CONST_MODE_HEAT
        )
        await self.coordinator.async_request_refresh()

    @callback
    def _async_update_callback(self) -> None:
        """Load tado data and update state."""
        self._async_update_data()
        self.async_write_ha_state()

    @callback
    def _async_update_data(self) -> None:
        """Load tado data."""
        _LOGGER.debug("Updating water_heater platform for zone %d", self.zone_id)
        self._tado_zone_data = self.coordinator.data["zone"][self.zone_id]
        self._current_tado_hvac_mode = self._tado_zone_data.current_hvac_mode

    async def _control_heater(
        self,
        hvac_mode: str | None = None,
        target_temp: float | None = None,
        duration: float | None = None,
    ) -> None:
        """Send new target temperature."""
        if hvac_mode:
            self._current_tado_hvac_mode = hvac_mode
        if target_temp:
            self._target_temp = target_temp
        if self._target_temp is None and self.min_temp is not None:
            self._target_temp = self.min_temp
        if self._current_tado_hvac_mode == CONST_MODE_SMART_SCHEDULE:
            _LOGGER.debug(
                "Switching to SMART_SCHEDULE for zone %s (%d)",
                self.zone_name,
                self.zone_id,
            )
            await self.coordinator.reset_zone_overlay(self.zone_id)
            await self.coordinator.async_request_refresh()
            return
        if self._current_tado_hvac_mode == CONST_MODE_OFF:
            _LOGGER.debug(
                "Switching to OFF for zone %s (%d)",
                self.zone_name,
                self.zone_id,
            )
            await self.coordinator.set_zone_off(
                self.zone_id, CONST_OVERLAY_MANUAL, TYPE_HOT_WATER
            )
            return
        overlay_mode = decide_overlay_mode(
            coordinator=self.coordinator, duration=duration, zone_id=self.zone_id
        )
        duration = decide_duration(
            coordinator=self.coordinator,
            duration=duration,
            zone_id=self.zone_id,
            overlay_mode=overlay_mode,
        )
        _LOGGER.debug(
            "Switching to %s for zone %s (%d) with temperature %s",
            self._current_tado_hvac_mode,
            self.zone_name,
            self.zone_id,
            self._target_temp,
        )
        await self.coordinator.set_zone_overlay(
            zone_id=self.zone_id,
            overlay_mode=overlay_mode,
            temperature=self._target_temp,
            duration=duration,
            device_type=TYPE_HOT_WATER,
        )
        self._overlay_mode = self._current_tado_hvac_mode
