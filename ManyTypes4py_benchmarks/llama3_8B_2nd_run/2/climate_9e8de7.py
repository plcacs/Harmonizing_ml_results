from __future__ import annotations
import logging
from typing import Any, Final, cast
from aiolookin import Climate, MeteoSensor, Remote
from aiolookin.models import UDPCommandType, UDPEvent
from homeassistant.components.climate import ATTR_HVAC_MODE, FAN_AUTO, FAN_HIGH, FAN_LOW, FAN_MIDDLE, SWING_BOTH, SWING_OFF, ClimateEntity, ClimateEntityFeature, HVACMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, PRECISION_WHOLE, Platform, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN, TYPE_TO_PLATFORM
from .coordinator import LookinDataUpdateCoordinator
from .entity import LookinCoordinatorEntity
from .models import LookinData

LOOKIN_FAN_MODE_IDX_TO_HASS: list[FAN_AUTO] = [FAN_AUTO, FAN_LOW, FAN_MIDDLE, FAN_HIGH]
LOOKIN_SWING_MODE_IDX_TO_HASS: list[SWING_BOTH] = [SWING_OFF, SWING_BOTH]
LOOKIN_HVAC_MODE_IDX_TO_HASS: list[HVACMode] = [HVACMode.OFF, HVACMode.AUTO, HVACMode.COOL, HVACMode.HEAT, HVACMode.DRY, HVACMode.FAN_ONLY]
HASS_TO_LOOKIN_HVAC_MODE: dict[HVACMode, int] = {mode: idx for idx, mode in enumerate(LOOKIN_HVAC_MODE_IDX_TO_HASS)}
HASS_TO_LOOKIN_FAN_MODE: dict[FAN_AUTO, int] = {mode: idx for idx, mode in enumerate(LOOKIN_FAN_MODE_IDX_TO_HASS)}
HASS_TO_LOOKIN_SWING_MODE: dict[SWING_BOTH, int] = {mode: idx for idx, mode in enumerate(LOOKIN_SWING_MODE_IDX_TO_HASS)}
MIN_TEMP: int = 16
MAX_TEMP: int = 30
LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ConditionerEntity(LookinCoordinatorEntity, ClimateEntity):
    _attr_current_humidity: int | None
    _attr_temperature_unit: UnitOfTemperature
    _attr_supported_features: ClimateEntityFeature
    _attr_fan_modes: list[FAN_AUTO]
    _attr_swing_modes: list[SWING_BOTH]
    _attr_hvac_modes: list[HVACMode]
    _attr_min_temp: int
    _attr_max_temp: int
    _attr_target_temperature_step: int

    def __init__(self, uuid: str, device: Climate, lookin_data: LookinData, coordinator: LookinDataUpdateCoordinator) -> None:
        ...

    @property
    def _climate(self) -> Climate:
        ...

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

    async def async_set_fan_mode(self, fan_mode: FAN_AUTO) -> None:
        ...

    async def async_set_swing_mode(self, swing_mode: SWING_BOTH) -> None:
        ...

    async def _async_update_conditioner(self) -> None:
        ...

    def _async_update_from_data(self) -> None:
        ...

    @callback
    def _async_update_meteo_from_value(self, event: UDPEvent) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    @callback
    def _async_push_update(self, event: UDPEvent) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...
