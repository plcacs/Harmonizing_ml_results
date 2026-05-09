from homeassistant.core import callback
from homeassistant.helpers.typing import VolDictType
from homeassistant.const import UnitOfTemperature
from typing import Any

class TadoWaterHeater(TadoZoneEntity, WaterHeaterEntity):
    """Representation of a Tado water heater."""
    _attr_name: str
    _attr_operation_list: list[str]
    _attr_temperature_unit: UnitOfTemperature

    def __init__(self, coordinator: Any, zone_name: str, zone_id: str, supports_temperature_control: bool, min_temp: float, max_temp: float) -> None:
        """Initialize of Tado water heater entity."""
        super().__init__(zone_name, coordinator.home_id, zone_id, coordinator)
        self.zone_id: str
        self._attr_unique_id: str
        self._device_is_active: bool
        self._supports_temperature_control: bool
        self._min_temperature: float
        self._max_temperature: float
        self._target_temp: float | None
        self._attr_supported_features: WaterHeaterEntityFeature
        self._current_tado_hvac_mode: str
        self._overlay_mode: str
        self._tado_zone_data: Any | None
        self._async_update_data()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self._async_update_data()
        super()._handle_coordinator_update()

    @property
    def current_operation(self) -> str:
        """Return current readable operation mode."""
        return WATER_HEATER_MAP_TADO.get(self._current_tado_hvac_mode)

    @property
    def target_temperature(self) -> float | None:
        """Return the temperature we try to reach."""
        return self._tado_zone_data.target_temp

    @property
    def is_away_mode_on(self) -> bool:
        """Return true if away mode is on."""
        return self._tado_zone_data.is_away

    @property
    def min_temp(self) -> float:
        """Return the minimum temperature."""
        return self._min_temperature

    @property
    def max_temp(self) -> float:
        """Return the maximum temperature."""
        return self._max_temperature

    async def async_set_operation_mode(self, operation_mode: str) -> None:
        """Set new operation mode."""
        mode: str | None
        if operation_mode == MODE_OFF:
            mode = CONST_MODE_OFF
        elif operation_mode == MODE_AUTO:
            mode = CONST_MODE_SMART_SCHEDULE
        elif operation_mode == MODE_HEAT:
            mode = CONST_MODE_HEAT
        await self._control_heater(hvac_mode=mode)
        await self.coordinator.async_request_refresh()

    async def set_timer(self, time_period: str, temperature: float | None = None) -> None:
        """Set the timer on the entity, and temperature if supported."""
        if not self._supports_temperature_control and temperature is not None:
            temperature = None
        await self._control_heater(hvac_mode=CONST_MODE_HEAT, target_temp=temperature, duration=time_period)
        await self.coordinator.async_request_refresh()

    async def async_set_temperature(self, **kwargs: VolDictType) -> None:
        """Set new target temperature."""
        temperature: float | None
        if not self._supports_temperature_control or (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        if self._current_tado_hvac_mode not in (CONST_MODE_OFF, CONST_MODE_AUTO, CONST_MODE_SMART_SCHEDULE):
            await self._control_heater(target_temp=temperature)
            return
        await self._control_heater(target_temp=temperature, hvac_mode=CONST_MODE_HEAT)
        await self.coordinator.async_request_refresh()

    @callback
    def _async_update_callback(self) -> None:
        """Load tado data and update state."""
        self._async_update_data()
        self.async_write_ha_state()

    @callback
    def _async_update_data(self) -> None:
        """Load tado data."""
        _LOGGER.debug('Updating water_heater platform for zone %d', self.zone_id)
        self._tado_zone_data = self.coordinator.data['zone'][self.zone_id]
        self._current_tado_hvac_mode = self._tado_zone_data.current_hvac_mode

    async def _control_heater(self, hvac_mode: str | None, target_temp: float | None, duration: str | None) -> None:
        """Send new target temperature."""
        if hvac_mode:
            self._current_tado_hvac_mode = hvac_mode
        if target_temp:
            self._target_temp = target_temp
        if self._target_temp is None:
            self._target_temp = self.min_temp
        if self._current_tado_hvac_mode == CONST_MODE_SMART_SCHEDULE:
            _LOGGER.debug('Switching to SMART_SCHEDULE for zone %s (%d)', self.zone_name, self.zone_id)
            await self.coordinator.reset_zone_overlay(self.zone_id)
            await self.coordinator.async_request_refresh()
            return
        if self._current_tado_hvac_mode == CONST_MODE_OFF:
            _LOGGER.debug('Switching to OFF for zone %s (%d)', self.zone_name, self.zone_id)
            await self.coordinator.set_zone_off(self.zone_id, CONST_OVERLAY_MANUAL, TYPE_HOT_WATER)
            return
        overlay_mode = decide_overlay_mode(coordinator=self.coordinator, duration=duration, zone_id=self.zone_id)
        duration = decide_duration(coordinator=self.coordinator, duration=duration, zone_id=self.zone_id, overlay_mode=overlay_mode)
        _LOGGER.debug('Switching to %s for zone %s (%d) with temperature %s', self._current_tado_hvac_mode, self.zone_name, self.zone_id, self._target_temp)
        await self.coordinator.set_zone_overlay(zone_id=self.zone_id, overlay_mode=overlay_mode, temperature=self._target_temp, duration=duration, device_type=TYPE_HOT_WATER)
        self._overlay_mode = self._current_tado_hvac_mode