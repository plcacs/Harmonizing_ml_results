async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
    """Set up platform."""
    ...

class NibeClimateEntity(CoordinatorEntity[CoilCoordinator], ClimateEntity):
    """Climate entity."""
    _attr_entity_category: str | None
    _attr_supported_features: ClimateEntityFeature
    _attr_hvac_modes: list[HVACMode]
    _attr_target_temperature_step: float
    _attr_max_temp: float
    _attr_min_temp: float

    def __init__(self, coordinator: CoilCoordinator, key: str, unit: UNIT_COILGROUPS, climate: ClimateCoilGroup):
        """Initialize entity."""
        super().__init__(coordinator, {unit.prio, unit.cooling_with_room_sensor, climate.current, climate.setpoint_heat, climate.setpoint_cool, climate.mixing_valve_state, climate.active_accessory, climate.use_room_sensor})
        ...

    @callback
    def _handle_coordinator_update(self):
        ...

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set target temperatures."""
        ...

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set new target hvac mode."""
        ...
