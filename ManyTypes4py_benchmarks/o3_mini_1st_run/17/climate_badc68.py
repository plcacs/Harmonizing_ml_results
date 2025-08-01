from __future__ import annotations
import logging
from typing import Any, Callable, Awaitable, Dict, List
from aiohttp import ClientSession
from whirlpool.aircon import Aircon, FanSpeed as AirconFanSpeed, Mode as AirconMode
from whirlpool.auth import Auth
from whirlpool.backendselector import BackendSelector
from homeassistant.components.climate import (
    ENTITY_ID_FORMAT,
    FAN_AUTO,
    FAN_HIGH,
    FAN_LOW,
    FAN_MEDIUM,
    FAN_OFF,
    SWING_HORIZONTAL,
    SWING_OFF,
    ClimateEntity,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import generate_entity_id
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import WhirlpoolConfigEntry
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

AIRCON_MODE_MAP: Dict[AirconMode, HVACMode] = {
    AirconMode.Cool: HVACMode.COOL,
    AirconMode.Heat: HVACMode.HEAT,
    AirconMode.Fan: HVACMode.FAN_ONLY,
}
HVAC_MODE_TO_AIRCON_MODE: Dict[HVACMode, AirconMode] = {v: k for k, v in AIRCON_MODE_MAP.items()}
AIRCON_FANSPEED_MAP: Dict[AirconFanSpeed, str] = {
    AirconFanSpeed.Off: FAN_OFF,
    AirconFanSpeed.Auto: FAN_AUTO,
    AirconFanSpeed.Low: FAN_LOW,
    AirconFanSpeed.Medium: FAN_MEDIUM,
    AirconFanSpeed.High: FAN_HIGH,
}
FAN_MODE_TO_AIRCON_FANSPEED: Dict[str, AirconFanSpeed] = {v: k for k, v in AIRCON_FANSPEED_MAP.items()}

SUPPORTED_FAN_MODES: List[str] = [FAN_AUTO, FAN_HIGH, FAN_MEDIUM, FAN_LOW, FAN_OFF]
SUPPORTED_HVAC_MODES: List[str] = [HVACMode.COOL, HVACMode.HEAT, HVACMode.FAN_ONLY, HVACMode.OFF]
SUPPORTED_MAX_TEMP: int = 30
SUPPORTED_MIN_TEMP: int = 16
SUPPORTED_SWING_MODES: List[str] = [SWING_HORIZONTAL, SWING_OFF]
SUPPORTED_TARGET_TEMPERATURE_STEP: int = 1


async def async_setup_entry(
    hass: HomeAssistant, 
    config_entry: WhirlpoolConfigEntry, 
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    whirlpool_data: Any = config_entry.runtime_data
    aircons = [
        AirConEntity(
            hass,
            ac_data["SAID"],
            ac_data["NAME"],
            whirlpool_data.backend_selector,
            whirlpool_data.auth,
            async_get_clientsession(hass),
        )
        for ac_data in whirlpool_data.appliances_manager.aircons
    ]
    async_add_entities(aircons, True)


class AirConEntity(ClimateEntity):
    _attr_fan_modes: List[str] = SUPPORTED_FAN_MODES
    _attr_has_entity_name: bool = True
    _attr_name: Any = None
    _attr_hvac_modes: List[str] = SUPPORTED_HVAC_MODES
    _attr_max_temp: int = SUPPORTED_MAX_TEMP
    _attr_min_temp: int = SUPPORTED_MIN_TEMP
    _attr_should_poll: bool = False
    _attr_supported_features: int = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.FAN_MODE
        | ClimateEntityFeature.SWING_MODE
        | ClimateEntityFeature.TURN_OFF
        | ClimateEntityFeature.TURN_ON
    )
    _attr_swing_modes: List[str] = SUPPORTED_SWING_MODES
    _attr_target_temperature_step: int = SUPPORTED_TARGET_TEMPERATURE_STEP
    _attr_temperature_unit: str = UnitOfTemperature.CELSIUS

    def __init__(
        self, 
        hass: HomeAssistant, 
        said: str, 
        name: str, 
        backend_selector: BackendSelector, 
        auth: Auth, 
        session: ClientSession
    ) -> None:
        self._aircon: Aircon = Aircon(backend_selector, auth, said, session)
        self.entity_id: str = generate_entity_id(ENTITY_ID_FORMAT, said, hass=hass)
        self._attr_unique_id: str = said
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, said)},
            name=name if name is not None else said,
            manufacturer="Whirlpool",
            model="Sixth Sense",
        )

    async def async_added_to_hass(self) -> None:
        self._aircon.register_attr_callback(self.async_write_ha_state)
        await self._aircon.connect()

    async def async_will_remove_from_hass(self) -> None:
        self._aircon.unregister_attr_callback(self.async_write_ha_state)
        await self._aircon.disconnect()

    @property
    def available(self) -> bool:
        return self._aircon.get_online()

    @property
    def current_temperature(self) -> float:
        return self._aircon.get_current_temp()

    @property
    def target_temperature(self) -> float:
        return self._aircon.get_temp()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        temp: Any = kwargs.get(ATTR_TEMPERATURE)
        await self._aircon.set_temp(temp)

    @property
    def current_humidity(self) -> float:
        return self._aircon.get_current_humidity()

    @property
    def target_humidity(self) -> float:
        return self._aircon.get_humidity()

    async def async_set_humidity(self, humidity: float) -> None:
        await self._aircon.set_humidity(humidity)

    @property
    def hvac_mode(self) -> str:
        if not self._aircon.get_power_on():
            return HVACMode.OFF
        mode: AirconMode = self._aircon.get_mode()
        return AIRCON_MODE_MAP.get(mode)

    async def async_set_hvac_mode(self, hvac_mode: str) -> None:
        if hvac_mode == HVACMode.OFF:
            await self._aircon.set_power_on(False)
            return
        mode: AirconMode | None = HVAC_MODE_TO_AIRCON_MODE.get(hvac_mode)
        if mode is None:
            raise ValueError(f"Invalid hvac mode {hvac_mode}")
        await self._aircon.set_mode(mode)
        if not self._aircon.get_power_on():
            await self._aircon.set_power_on(True)

    @property
    def fan_mode(self) -> str:
        fanspeed: AirconFanSpeed = self._aircon.get_fanspeed()
        return AIRCON_FANSPEED_MAP.get(fanspeed, FAN_OFF)

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        fanspeed: AirconFanSpeed | None = FAN_MODE_TO_AIRCON_FANSPEED.get(fan_mode)
        if fanspeed is None:
            raise ValueError(f"Invalid fan mode {fan_mode}")
        await self._aircon.set_fanspeed(fanspeed)

    @property
    def swing_mode(self) -> str:
        return SWING_HORIZONTAL if self._aircon.get_h_louver_swing() else SWING_OFF

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        await self._aircon.set_h_louver_swing(swing_mode == SWING_HORIZONTAL)

    async def async_turn_on(self) -> None:
        await self._aircon.set_power_on(True)

    async def async_turn_off(self) -> None:
        await self._aircon.set_power_on(False)