from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
from homeassistant.components.climate import ClimateEntity, ClimateEntityDescription, ClimateEntityFeature, HVACMode
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import TUYA_DISCOVERY_NEW, DPCode, DPType
from .entity import IntegerTypeData, TuyaEntity

@dataclass(frozen=True, kw_only=True)
class TuyaClimateEntityDescription(ClimateEntityDescription):
    """Describe an Tuya climate entity."""

async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up Tuya climate dynamically through Tuya discovery."""

class TuyaClimateEntity(TuyaEntity, ClimateEntity):
    """Tuya Climate Device."""
    _current_humidity: Optional[IntegerTypeData] = None
    _current_temperature: Optional[IntegerTypeData] = None
    _set_humidity: Optional[IntegerTypeData] = None
    _set_temperature: Optional[IntegerTypeData] = None
    _attr_name: Optional[str] = None

    def __init__(self, device: CustomerDevice, device_manager: Manager, description: TuyaClimateEntityDescription, system_temperature_unit: UnitOfTemperature) -> None:
        """Determine which values to use."""

    async def async_added_to_hass(self) -> None:
        """Call when entity is added to hass."""

    def set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set new target hvac mode."""

    def set_preset_mode(self, preset_mode: str) -> None:
        """Set new target preset mode."""

    def set_fan_mode(self, fan_mode: str) -> None:
        """Set new target fan mode."""

    def set_humidity(self, humidity: int) -> None:
        """Set new target humidity."""

    def set_swing_mode(self, swing_mode: str) -> None:
        """Set new target swing operation."""

    def set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""

    @property
    def current_temperature(self) -> Optional[float]:
        """Return the current temperature."""

    @property
    def current_humidity(self) -> Optional[int]:
        """Return the current humidity."""

    @property
    def target_temperature(self) -> Optional[float]:
        """Return the temperature currently set to be reached."""

    @property
    def target_humidity(self) -> Optional[int]:
        """Return the humidity currently set to be reached."""

    @property
    def hvac_mode(self) -> HVACMode:
        """Return hvac mode."""

    @property
    def preset_mode(self) -> Optional[str]:
        """Return preset mode."""

    @property
    def fan_mode(self) -> Optional[str]:
        """Return fan mode."""

    @property
    def swing_mode(self) -> str:
        """Return swing mode."""

    def turn_on(self) -> None:
        """Turn the device on, retaining current HVAC (if supported)."""

    def turn_off(self) -> None:
        """Turn the device on, retaining current HVAC (if supported)."""
