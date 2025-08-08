from __future__ import annotations
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class DemoClimate(ClimateEntity):
    def __init__(self, unique_id: str, device_name: str, target_temperature: float, unit_of_measurement: UnitOfTemperature, preset: str, current_temperature: float, fan_mode: str, target_humidity: float, current_humidity: float, swing_mode: str, swing_horizontal_mode: str, hvac_mode: HVACMode, hvac_action: HVACAction, target_temp_high: float, target_temp_low: float, hvac_modes: list[HVACMode], preset_modes: list[str] = None) -> None:

    @property
    def unique_id(self) -> str:
    @property
    def temperature_unit(self) -> UnitOfTemperature:
    @property
    def current_temperature(self) -> float:
    @property
    def target_temperature(self) -> float:
    @property
    def target_temperature_high(self) -> float:
    @property
    def target_temperature_low(self) -> float:
    @property
    def current_humidity(self) -> float:
    @property
    def target_humidity(self) -> float:
    @property
    def hvac_action(self) -> HVACAction:
    @property
    def hvac_mode(self) -> HVACMode:
    @property
    def hvac_modes(self) -> list[HVACMode]:
    @property
    def preset_mode(self) -> str:
    @property
    def preset_modes(self) -> list[str]:
    @property
    def fan_mode(self) -> str:
    @property
    def fan_modes(self) -> list[str]:
    @property
    def swing_mode(self) -> str:
    @property
    def swing_modes(self) -> list[str]:
    @property
    def swing_horizontal_mode(self) -> str:
    @property
    def swing_horizontal_modes(self) -> list[str]:

    async def async_set_temperature(self, **kwargs: Any) -> None:
    async def async_set_humidity(self, humidity: float) -> None:
    async def async_set_swing_mode(self, swing_mode: str) -> None:
    async def async_set_swing_horizontal_mode(self, swing_horizontal_mode: str) -> None:
    async def async_set_fan_mode(self, fan_mode: str) -> None:
    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
    async def async_set_preset_mode(self, preset_mode: str) -> None:
