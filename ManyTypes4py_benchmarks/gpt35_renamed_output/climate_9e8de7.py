from __future__ import annotations
import logging
from typing import Any, Final, cast, List, Dict, Union
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

LOOKIN_FAN_MODE_IDX_TO_HASS: List[str] = [FAN_AUTO, FAN_LOW, FAN_MIDDLE, FAN_HIGH]
LOOKIN_SWING_MODE_IDX_TO_HASS: List[str] = [SWING_OFF, SWING_BOTH]
LOOKIN_HVAC_MODE_IDX_TO_HASS: List[HVACMode] = [HVACMode.OFF, HVACMode.AUTO, HVACMode.COOL, HVACMode.HEAT, HVACMode.DRY, HVACMode.FAN_ONLY]
HASS_TO_LOOKIN_HVAC_MODE: Dict[HVACMode, int] = {mode: idx for idx, mode in enumerate(LOOKIN_HVAC_MODE_IDX_TO_HASS)}
HASS_TO_LOOKIN_FAN_MODE: Dict[str, int] = {mode: idx for idx, mode in enumerate(LOOKIN_FAN_MODE_IDX_TO_HASS)}
HASS_TO_LOOKIN_SWING_MODE: Dict[str, int] = {mode: idx for idx, mode in enumerate(LOOKIN_SWING_MODE_IDX_TO_HASS)}

MIN_TEMP: Final[int] = 16
MAX_TEMP: Final[int] = 30
LOGGER: logging.Logger = logging.getLogger(__name__)

async def func_i3whlj7i(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ConditionerEntity(LookinCoordinatorEntity, ClimateEntity):
    _attr_current_humidity: Union[float, None] = None
    _attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS
    _attr_supported_features: int = (ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.FAN_MODE | ClimateEntityFeature.SWING_MODE | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON)
    _attr_fan_modes: List[str] = LOOKIN_FAN_MODE_IDX_TO_HASS
    _attr_swing_modes: List[str] = LOOKIN_SWING_MODE_IDX_TO_HASS
    _attr_hvac_modes: List[HVACMode] = LOOKIN_HVAC_MODE_IDX_TO_HASS
    _attr_min_temp: int = MIN_TEMP
    _attr_max_temp: int = MAX_TEMP
    _attr_target_temperature_step: float = PRECISION_WHOLE

    def __init__(self, uuid: str, device: Climate, lookin_data: LookinData, coordinator: LookinDataUpdateCoordinator) -> None:
        ...

    @property
    def func_tfq8zf2i(self) -> Climate:
        ...

    async def func_xhwp0pvc(self, hvac_mode: str) -> None:
        ...

    async def func_ayqhkogx(self, **kwargs: Any) -> None:
        ...

    async def func_8k0tz4ez(self, fan_mode: str) -> None:
        ...

    async def func_ujquf65q(self, swing_mode: str) -> None:
        ...

    async def func_u4gn7knz(self) -> None:
        ...

    def func_jex6yp0a(self) -> None:
        ...

    @callback
    def func_ha05q4qx(self, event: UDPEvent) -> None:
        ...

    @callback
    def func_z5dpi8vo(self) -> None:
        ...

    @callback
    def func_kapd5fsh(self, event: UDPEvent) -> None:
        ...

    async def func_jd8lhqle(self) -> None:
        ...
