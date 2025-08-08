from __future__ import annotations
from datetime import date
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class NibeClimateEntity(CoordinatorEntity[CoilCoordinator], ClimateEntity):
    _attr_supported_features: int = ClimateEntityFeature.TARGET_TEMPERATURE_RANGE | ClimateEntityFeature.TARGET_TEMPERATURE
    _attr_hvac_modes: list[HVACMode] = [HVACMode.AUTO, HVACMode.HEAT, HVACMode.HEAT_COOL]
    _attr_target_temperature_step: float = 0.5
    _attr_max_temp: float = 35.0
    _attr_min_temp: float = 5.0

    def __init__(self, coordinator: CoilCoordinator, key: str, unit: UnitCoilGroup, climate: ClimateCoilGroup) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    @property
    def available(self) -> bool:
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        ...
