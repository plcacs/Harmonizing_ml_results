"""The Nibe Heat Pump climate."""
from __future__ import annotations
from datetime import date
from typing import Any, Generator, Callable
from nibe.coil import Coil
from nibe.coil_groups import CLIMATE_COILGROUPS, UNIT_COILGROUPS, ClimateCoilGroup, UnitCoilGroup
from nibe.exceptions import CoilNotFoundException
from homeassistant.components.climate import (
    ATTR_HVAC_MODE,
    ATTR_TARGET_TEMP_HIGH,
    ATTR_TARGET_TEMP_LOW,
    ATTR_TEMPERATURE,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import (
    DOMAIN,
    LOGGER,
    VALUES_COOL_WITH_ROOM_SENSOR_OFF,
    VALUES_MIXING_VALVE_CLOSED_STATE,
    VALUES_PRIORITY_COOLING,
    VALUES_PRIORITY_HEATING,
)
from .coordinator import CoilCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up platform."""
    coordinator: CoilCoordinator = hass.data[DOMAIN][config_entry.entry_id]
    main_unit: UnitCoilGroup = UNIT_COILGROUPS[coordinator.series]["main"]

    def climate_systems() -> Generator[ClimateEntity, None, None]:
        for key, group in CLIMATE_COILGROUPS.get(coordinator.series, {}).items():
            try:
                yield NibeClimateEntity(coordinator, key, main_unit, group)
            except CoilNotFoundException as exception:
                LOGGER.debug("Skipping climate: %s due to %s", key, exception)

    async_add_entities(climate_systems())


class NibeClimateEntity(CoordinatorEntity[CoilCoordinator], ClimateEntity):
    """Climate entity."""
    _attr_entity_category = None
    _attr_supported_features: int = ClimateEntityFeature.TARGET_TEMPERATURE_RANGE | ClimateEntityFeature.TARGET_TEMPERATURE
    _attr_hvac_modes: list[str] = [HVACMode.AUTO, HVACMode.HEAT, HVACMode.HEAT_COOL]
    _attr_target_temperature_step: float = 0.5
    _attr_max_temp: float = 35.0
    _attr_min_temp: float = 5.0

    def __init__(
        self,
        coordinator: CoilCoordinator,
        key: str,
        unit: UnitCoilGroup,
        climate: ClimateCoilGroup,
    ) -> None:
        """Initialize entity."""
        super().__init__(
            coordinator,
            {
                unit.prio,
                unit.cooling_with_room_sensor,
                climate.current,
                climate.setpoint_heat,
                climate.setpoint_cool,
                climate.mixing_valve_state,
                climate.active_accessory,
                climate.use_room_sensor,
            },
        )
        self._attr_available: bool = False
        self._attr_name: str = climate.name
        self._attr_unique_id: str = f"{coordinator.unique_id}-{key}"
        self._attr_device_info = coordinator.device_info
        self._attr_hvac_action: str = HVACAction.IDLE
        self._attr_hvac_mode: str = HVACMode.AUTO
        self._attr_target_temperature_high: float | None = None
        self._attr_target_temperature_low: float | None = None
        self._attr_target_temperature: float | None = None
        self._attr_entity_registry_enabled_default: bool = climate.active_accessory is None

        def _get(address: str) -> Coil:
            return coordinator.heatpump.get_coil_by_address(address)

        self._coil_current: Coil = _get(climate.current)
        self._coil_setpoint_heat: Coil = _get(climate.setpoint_heat)
        try:
            self._coil_setpoint_cool: Coil | None = _get(climate.setpoint_cool)
        except CoilNotFoundException:
            self._coil_setpoint_cool = None
            self._attr_hvac_modes = [HVACMode.AUTO, HVACMode.HEAT]
        self._coil_prio: Coil = _get(unit.prio)
        self._coil_mixing_valve_state: Coil = _get(climate.mixing_valve_state)
        if climate.active_accessory is None:
            self._coil_active_accessory = None
        else:
            self._coil_active_accessory = _get(climate.active_accessory)
        self._coil_use_room_sensor: Coil = _get(climate.use_room_sensor)
        self._coil_cooling_with_room_sensor: Coil = _get(unit.cooling_with_room_sensor)
        if self._coil_current:
            self._attr_temperature_unit = self._coil_current.unit

    @callback
    def _handle_coordinator_update(self) -> None:

        def _get_value(coil: Coil) -> Any:
            return self.coordinator.get_coil_value(coil)

        def _get_float(coil: Coil) -> float:
            return self.coordinator.get_coil_float(coil)

        self._attr_current_temperature: float = _get_float(self._coil_current)
        mode: str = HVACMode.AUTO
        if _get_value(self._coil_use_room_sensor) == "ON":
            if _get_value(self._coil_cooling_with_room_sensor) in VALUES_COOL_WITH_ROOM_SENSOR_OFF:
                mode = HVACMode.HEAT
            else:
                mode = HVACMode.HEAT_COOL
        self._attr_hvac_mode = mode
        setpoint_heat: float = _get_float(self._coil_setpoint_heat)
        if self._coil_setpoint_cool:
            setpoint_cool: float = _get_float(self._coil_setpoint_cool)
        else:
            setpoint_cool = None
        if mode == HVACMode.HEAT_COOL:
            self._attr_target_temperature = None
            self._attr_target_temperature_low = setpoint_heat
            self._attr_target_temperature_high = setpoint_cool
        elif mode == HVACMode.HEAT:
            self._attr_target_temperature = setpoint_heat
            self._attr_target_temperature_low = None
            self._attr_target_temperature_high = None
        else:
            self._attr_target_temperature = None
            self._attr_target_temperature_low = None
            self._attr_target_temperature_high = None
        if (prio := _get_value(self._coil_prio)):
            if _get_value(self._coil_mixing_valve_state) in VALUES_MIXING_VALVE_CLOSED_STATE:
                self._attr_hvac_action = HVACAction.IDLE
            elif prio in VALUES_PRIORITY_HEATING:
                self._attr_hvac_action = HVACAction.HEATING
            elif prio in VALUES_PRIORITY_COOLING:
                self._attr_hvac_action = HVACAction.COOLING
            else:
                self._attr_hvac_action = HVACAction.IDLE
        else:
            self._attr_hvac_action = HVACAction.OFF
        self.async_write_ha_state()

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        coordinator: CoilCoordinator = self.coordinator
        active = self._coil_active_accessory
        if not coordinator.last_update_success:
            return False
        if not active:
            return True
        active_accessory: Any = coordinator.get_coil_value(active)
        return active_accessory == "ON"

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set target temperatures."""
        coordinator: CoilCoordinator = self.coordinator
        hvac_mode: str = kwargs.get(ATTR_HVAC_MODE, self._attr_hvac_mode)
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is not None:
            if hvac_mode == HVACMode.HEAT:
                await coordinator.async_write_coil(self._coil_setpoint_heat, temperature)
            elif hvac_mode == HVACMode.COOL:
                if self._coil_setpoint_cool:
                    await coordinator.async_write_coil(self._coil_setpoint_cool, temperature)
                else:
                    raise ServiceValidationError(f"{hvac_mode} mode not supported for {self.name}")
            else:
                raise ServiceValidationError(
                    "'set_temperature' requires 'hvac_mode' when passing 'temperature' and 'hvac_mode' is not already set to 'heat' or 'cool'"
                )
        if (temperature := kwargs.get(ATTR_TARGET_TEMP_LOW)) is not None:
            await coordinator.async_write_coil(self._coil_setpoint_heat, temperature)
        if self._coil_setpoint_cool and (temperature := kwargs.get(ATTR_TARGET_TEMP_HIGH)) is not None:
            await coordinator.async_write_coil(self._coil_setpoint_cool, temperature)

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        """Set new target hvac mode."""
        coordinator: CoilCoordinator = self.coordinator
        if hvac_mode == HVACMode.HEAT_COOL:
            await coordinator.async_write_coil(self._coil_cooling_with_room_sensor, "ON")
            await coordinator.async_write_coil(self._coil_use_room_sensor, "ON")
        elif hvac_mode == HVACMode.HEAT:
            await coordinator.async_write_coil(self._coil_cooling_with_room_sensor, "OFF")
            await coordinator.async_write_coil(self._coil_use_room_sensor, "ON")
        elif hvac_mode == HVACMode.AUTO:
            await coordinator.async_write_coil(self._coil_cooling_with_room_sensor, "OFF")
            await coordinator.async_write_coil(self._coil_use_room_sensor, "OFF")
        else:
            raise ServiceValidationError(f"{hvac_mode} mode not supported for {self.name}")