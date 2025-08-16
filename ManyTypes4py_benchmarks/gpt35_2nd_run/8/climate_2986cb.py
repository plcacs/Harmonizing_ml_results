from __future__ import annotations
from typing import Any

async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

def _async_migrate_unique_id(hass: HomeAssistant, devices: dict[str, SomeComfortDevice]) -> None:
    ...

def remove_stale_devices(hass: HomeAssistant, config_entry: HoneywellConfigEntry, devices: dict[str, SomeComfortDevice]) -> None:
    ...

class HoneywellUSThermostat(ClimateEntity):
    def __init__(self, data: HoneywellData, device: SomeComfortDevice, cool_away_temp: float, heat_away_temp: float) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    @property
    def min_temp(self) -> float:
        ...

    @property
    def max_temp(self) -> float:
        ...

    @property
    def current_humidity(self) -> float:
        ...

    @property
    def hvac_mode(self) -> HVACMode:
        ...

    @property
    def hvac_action(self) -> HVACAction:
        ...

    @property
    def current_temperature(self) -> float:
        ...

    @property
    def target_temperature(self) -> float:
        ...

    @property
    def target_temperature_high(self) -> float:
        ...

    @property
    def target_temperature_low(self) -> float:
        ...

    @property
    def preset_mode(self) -> str:
        ...

    @property
    def fan_mode(self) -> str:
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        ...

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        ...

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        ...

    async def async_update(self) -> None:
        ...
