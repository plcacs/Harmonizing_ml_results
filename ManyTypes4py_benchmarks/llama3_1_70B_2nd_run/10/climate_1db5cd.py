from __future__ import annotations
from typing import Any, Final, Dict, List, Optional, Set

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Add Airzone climate from a config_entry."""
    coordinator: AirzoneUpdateCoordinator = entry.runtime_data
    added_zones: Set[str] = set()

    def _async_entity_listener() -> None:
        """Handle additions of climate."""
        zones_data: Dict[str, Any] = coordinator.data.get(AZD_ZONES, {})
        received_zones: Set[str] = set(zones_data)
        new_zones: Set[str] = received_zones - added_zones
        if new_zones:
            async_add_entities((AirzoneClimate(coordinator, entry, system_zone_id, zones_data.get(system_zone_id)) for system_zone_id in new_zones))
            added_zones.update(new_zones)
    entry.async_on_unload(coordinator.async_add_listener(_async_entity_listener))
    _async_entity_listener()

class AirzoneClimate(AirzoneZoneEntity, ClimateEntity):
    """Define an Airzone sensor."""
    _attr_name: Optional[str] = None
    _speeds: Dict[int, str] = {}
    _speeds_reverse: Dict[str, int] = {}

    def __init__(self, coordinator: AirzoneUpdateCoordinator, entry: ConfigEntry, system_zone_id: str, zone_data: Any) -> None:
        """Initialize Airzone climate entity."""
        super().__init__(coordinator, entry, system_zone_id, zone_data)
        self._attr_unique_id: str = f'{self._attr_unique_id}_{system_zone_id}'
        self._attr_supported_features: int = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
        self._attr_target_temperature_step: float = API_TEMPERATURE_STEP
        self._attr_temperature_unit: str = TEMP_UNIT_LIB_TO_HASS[self.get_airzone_value(AZD_TEMP_UNIT)]
        _attr_hvac_modes: List[HVACMode] = [HVAC_MODE_LIB_TO_HASS[mode] for mode in self.get_airzone_value(AZD_MODES)]
        self._attr_hvac_modes: List[HVACMode] = list(dict.fromkeys(_attr_hvac_modes))
        if self.get_airzone_value(AZD_SPEED) is not None and self.get_airzone_value(AZD_SPEEDS) is not None:
            self._set_fan_speeds()
        if self.get_airzone_value(AZD_DOUBLE_SET_POINT):
            self._attr_supported_features |= ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
        self._async_update_attrs()

    def _set_fan_speeds(self) -> None:
        """Set fan speeds."""
        self._attr_supported_features |= ClimateEntityFeature.FAN_MODE
        speeds: List[int] = self.get_airzone_value(AZD_SPEEDS)
        max_speed: int = max(speeds)
        if (_speeds := FAN_SPEED_MAPS.get(max_speed)):
            self._speeds = _speeds
        else:
            for speed in speeds:
                if speed == 0:
                    self._speeds[speed] = FAN_AUTO
                else:
                    self._speeds[speed] = f'{int(round(speed * 100 / max_speed, 0))}%'
            self._speeds[1] = FAN_LOW
            self._speeds[int(round((max_speed + 1) / 2, 0))] = FAN_MEDIUM
            self._speeds[max_speed] = FAN_HIGH
        self._speeds_reverse: Dict[str, int] = {v: k for k, v in self._speeds.items()}
        self._attr_fan_modes: List[str] = list(self._speeds_reverse)

    async def async_turn_on(self) -> None:
        """Turn the entity on."""
        params: Dict[str, Any] = {API_ON: 1}
        await self._async_update_hvac_params(params)

    async def async_turn_off(self) -> None:
        """Turn the entity off."""
        params: Dict[str, Any] = {API_ON: 0}
        await self._async_update_hvac_params(params)

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Set fan mode."""
        params: Dict[str, Any] = {API_SPEED: self._speeds_reverse.get(fan_mode)}
        await self._async_update_hvac_params(params)

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        slave_raise: bool = False
        params: Dict[str, Any] = {}
        if hvac_mode == HVACMode.OFF:
            params[API_ON] = 0
        else:
            mode: OperationMode = HVAC_MODE_HASS_TO_LIB[hvac_mode]
            if mode != self.get_airzone_value(AZD_MODE):
                if self.get_airzone_value(AZD_MASTER):
                    params[API_MODE] = mode
                else:
                    slave_raise = True
            params[API_ON] = 1
        await self._async_update_hvac_params(params)
        if slave_raise:
            raise HomeAssistantError(f"Mode can't be changed on slave zone {self.entity_id}")

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        params: Dict[str, Any] = {}
        if ATTR_TEMPERATURE in kwargs:
            params[API_SET_POINT] = kwargs[ATTR_TEMPERATURE]
        if ATTR_TARGET_TEMP_LOW in kwargs and ATTR_TARGET_TEMP_HIGH in kwargs:
            params[API_COOL_SET_POINT] = kwargs[ATTR_TARGET_TEMP_HIGH]
            params[API_HEAT_SET_POINT] = kwargs[ATTR_TARGET_TEMP_LOW]
        await self._async_update_hvac_params(params)
        if ATTR_HVAC_MODE in kwargs:
            await self.async_set_hvac_mode(kwargs[ATTR_HVAC_MODE])

    @callback
    def _handle_coordinator_update(self) -> None:
        """Update attributes when the coordinator updates."""
        self._async_update_attrs()
        super()._handle_coordinator_update()

    @callback
    def _async_update_attrs(self) -> None:
        """Update climate attributes."""
        self._attr_current_temperature: float = self.get_airzone_value(AZD_TEMP)
        self._attr_current_humidity: float = self.get_airzone_value(AZD_HUMIDITY)
        self._attr_hvac_action: HVACAction = HVAC_ACTION_LIB_TO_HASS[self.get_airzone_value(AZD_ACTION)]
        if self.get_airzone_value(AZD_ON):
            self._attr_hvac_mode: HVACMode = HVAC_MODE_LIB_TO_HASS[self.get_airzone_value(AZD_MODE)]
        else:
            self._attr_hvac_mode: HVACMode = HVACMode.OFF
        self._attr_max_temp: float = self.get_airzone_value(AZD_TEMP_MAX)
        self._attr_min_temp: float = self.get_airzone_value(AZD_TEMP_MIN)
        if self.supported_features & ClimateEntityFeature.FAN_MODE:
            self._attr_fan_mode: str = self._speeds.get(self.get_airzone_value(AZD_SPEED))
        if self.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE_RANGE and self._attr_hvac_mode == HVACMode.HEAT_COOL:
            self._attr_target_temperature_high: float = self.get_airzone_value(AZD_COOL_TEMP_SET)
            self._attr_target_temperature_low: float = self.get_airzone_value(AZD_HEAT_TEMP_SET)
            self._attr_target_temperature: Optional[float] = None
        else:
            self._attr_target_temperature_high: Optional[float] = None
            self._attr_target_temperature_low: Optional[float] = None
            self._attr_target_temperature: float = self.get_airzone_value(AZD_TEMP_SET)
