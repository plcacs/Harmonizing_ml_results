"""Support for Tuya Climate."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from tuya_sharing import CustomerDevice, Manager
from homeassistant.components.climate import (
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_ON,
    SWING_VERTICAL,
    ClimateEntity,
    ClimateEntityDescription,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import TuyaConfigEntry
from .const import TUYA_DISCOVERY_NEW, DPCode, DPType
from .entity import IntegerTypeData, TuyaEntity

TUYA_HVAC_TO_HA: Dict[str, HVACMode] = {
    'auto': HVACMode.HEAT_COOL,
    'cold': HVACMode.COOL,
    'freeze': HVACMode.COOL,
    'heat': HVACMode.HEAT,
    'hot': HVACMode.HEAT,
    'manual': HVACMode.HEAT_COOL,
    'wet': HVACMode.DRY,
    'wind': HVACMode.FAN_ONLY,
}

@dataclass(frozen=True, kw_only=True)
class TuyaClimateEntityDescription(ClimateEntityDescription):
    """Describe a Tuya climate entity."""

CLIMATE_DESCRIPTIONS: Dict[str, TuyaClimateEntityDescription] = {
    'kt': TuyaClimateEntityDescription(key='kt', switch_only_hvac_mode=HVACMode.COOL),
    'qn': TuyaClimateEntityDescription(key='qn', switch_only_hvac_mode=HVACMode.HEAT),
    'rs': TuyaClimateEntityDescription(key='rs', switch_only_hvac_mode=HVACMode.HEAT),
    'wk': TuyaClimateEntityDescription(key='wk', switch_only_hvac_mode=HVACMode.HEAT_COOL),
    'wkf': TuyaClimateEntityDescription(key='wkf', switch_only_hvac_mode=HVACMode.HEAT),
    'dbl': TuyaClimateEntityDescription(key='dbl', switch_only_hvac_mode=HVACMode.HEAT),
}

async def async_setup_entry(
    hass: HomeAssistant,
    entry: TuyaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Tuya climate dynamically through Tuya discovery."""
    hass_data = entry.runtime_data

    @callback
    def async_discover_device(device_ids: List[str]) -> None:
        """Discover and add a discovered Tuya climate."""
        entities: List[TuyaClimateEntity] = []
        for device_id in device_ids:
            device: Optional[CustomerDevice] = hass_data.manager.device_map.get(device_id)
            if device and device.category in CLIMATE_DESCRIPTIONS:
                entities.append(
                    TuyaClimateEntity(
                        device=device,
                        device_manager=hass_data.manager,
                        description=CLIMATE_DESCRIPTIONS[device.category],
                        system_temperature_unit=hass.config.units.temperature_unit,
                    )
                )
        async_add_entities(entities)

    async_discover_device(list(hass_data.manager.device_map))
    entry.async_on_unload(
        async_dispatcher_connect(hass, TUYA_DISCOVERY_NEW, async_discover_device)
    )

class TuyaClimateEntity(TuyaEntity, ClimateEntity):
    """Tuya Climate Device."""

    _current_humidity: Optional[IntegerTypeData]
    _current_temperature: Optional[IntegerTypeData]
    _set_humidity: Optional[IntegerTypeData]
    _set_temperature: Optional[IntegerTypeData]
    _attr_name: Optional[str]

    def __init__(
        self,
        device: CustomerDevice,
        device_manager: Manager,
        description: TuyaClimateEntityDescription,
        system_temperature_unit: str,
    ) -> None:
        """Determine which values to use."""
        self._attr_target_temperature_step: float = 1.0
        self.entity_description = description
        super().__init__(device, device_manager)
        prefered_temperature_unit: Optional[str] = None
        if all(
            (
                dpcode in device.status
                for dpcode in (DPCode.TEMP_CURRENT, DPCode.TEMP_CURRENT_F)
            )
        ) or all(
            (
                dpcode in device.status
                for dpcode in (DPCode.TEMP_SET, DPCode.TEMP_SET_F)
            )
        ):
            prefered_temperature_unit = UnitOfTemperature.CELSIUS
            if any(
                (
                    'f' in device.status.get(dpcode, '').lower()
                    for dpcode in (DPCode.C_F, DPCode.TEMP_UNIT_CONVERT)
                    if isinstance(device.status.get(dpcode), str)
                )
            ):
                prefered_temperature_unit = UnitOfTemperature.FAHRENHEIT
        self._attr_temperature_unit: str = system_temperature_unit
        celsius_type: Optional[IntegerTypeData] = self.find_dpcode(
            (DPCode.TEMP_CURRENT, DPCode.UPPER_TEMP), dptype=DPType.INTEGER
        )
        fahrenheit_type: Optional[IntegerTypeData] = self.find_dpcode(
            (DPCode.TEMP_CURRENT_F, DPCode.UPPER_TEMP_F), dptype=DPType.INTEGER
        )
        if (
            fahrenheit_type
            and (
                prefered_temperature_unit == UnitOfTemperature.FAHRENHEIT
                or (
                    prefered_temperature_unit == UnitOfTemperature.CELSIUS
                    and not celsius_type
                )
            )
        ):
            self._attr_temperature_unit = UnitOfTemperature.FAHRENHEIT
            self._current_temperature = fahrenheit_type
        elif celsius_type:
            self._attr_temperature_unit = UnitOfTemperature.CELSIUS
            self._current_temperature = celsius_type
        celsius_type = self.find_dpcode(
            DPCode.TEMP_SET, dptype=DPType.INTEGER, prefer_function=True
        )
        fahrenheit_type = self.find_dpcode(
            DPCode.TEMP_SET_F, dptype=DPType.INTEGER, prefer_function=True
        )
        if (
            fahrenheit_type
            and (
                prefered_temperature_unit == UnitOfTemperature.FAHRENHEIT
                or (
                    prefered_temperature_unit == UnitOfTemperature.CELSIUS
                    and not celsius_type
                )
            )
        ):
            self._set_temperature = fahrenheit_type
        elif celsius_type:
            self._set_temperature = celsius_type
        if self._set_temperature:
            self._attr_supported_features |= ClimateEntityFeature.TARGET_TEMPERATURE
            self._attr_max_temp = self._set_temperature.max_scaled
            self._attr_min_temp = self._set_temperature.min_scaled
            self._attr_target_temperature_step = self._set_temperature.step_scaled
        self._attr_hvac_modes: List[HVACMode] = []
        self._hvac_to_tuya: Dict[HVACMode, str] = {}
        enum_type: Optional[Any] = self.find_dpcode(
            DPCode.MODE, dptype=DPType.ENUM, prefer_function=True
        )
        if enum_type:
            self._attr_hvac_modes.append(HVACMode.OFF)
            unknown_hvac_modes: List[str] = []
            for tuya_mode in enum_type.range:
                if tuya_mode in TUYA_HVAC_TO_HA:
                    ha_mode: HVACMode = TUYA_HVAC_TO_HA[tuya_mode]
                    self._hvac_to_tuya[ha_mode] = tuya_mode
                    self._attr_hvac_modes.append(ha_mode)
                else:
                    unknown_hvac_modes.append(tuya_mode)
            if unknown_hvac_modes:
                self._attr_hvac_modes.append(description.switch_only_hvac_mode)
                self._attr_preset_modes = unknown_hvac_modes
                self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
        elif self.find_dpcode(DPCode.SWITCH, prefer_function=True):
            self._attr_hvac_modes = [HVACMode.OFF, description.switch_only_hvac_mode]
        int_type: Optional[IntegerTypeData] = self.find_dpcode(
            DPCode.HUMIDITY_SET, dptype=DPType.INTEGER, prefer_function=True
        )
        if int_type:
            self._attr_supported_features |= ClimateEntityFeature.TARGET_HUMIDITY
            self._set_humidity = int_type
            self._attr_min_humidity = int(int_type.min_scaled)
            self._attr_max_humidity = int(int_type.max_scaled)
        self._current_humidity = self.find_dpcode(
            DPCode.HUMIDITY_CURRENT, dptype=DPType.INTEGER
        )
        enum_type = self.find_dpcode(
            (DPCode.FAN_SPEED_ENUM, DPCode.WINDSPEED),
            dptype=DPType.ENUM,
            prefer_function=True,
        )
        if enum_type:
            self._attr_supported_features |= ClimateEntityFeature.FAN_MODE
            self._attr_fan_modes = enum_type.range
        if self.find_dpcode(
            (DPCode.SHAKE, DPCode.SWING, DPCode.SWITCH_HORIZONTAL, DPCode.SWITCH_VERTICAL),
            prefer_function=True,
        ):
            self._attr_supported_features |= ClimateEntityFeature.SWING_MODE
            self._attr_swing_modes = [SWING_OFF]
            if self.find_dpcode((DPCode.SHAKE, DPCode.SWING), prefer_function=True):
                self._attr_swing_modes.append(SWING_ON)
            if self.find_dpcode(DPCode.SWITCH_HORIZONTAL, prefer_function=True):
                self._attr_swing_modes.append(SWING_HORIZONTAL)
            if self.find_dpcode(DPCode.SWITCH_VERTICAL, prefer_function=True):
                self._attr_swing_modes.append(SWING_VERTICAL)
        if DPCode.SWITCH in self.device.function:
            self._attr_supported_features |= ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON

    async def async_added_to_hass(self) -> None:
        """Call when entity is added to hass."""
        await super().async_added_to_hass()

    def set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set new target hvac mode."""
        commands: List[Dict[str, Any]] = [{'code': DPCode.SWITCH, 'value': hvac_mode != HVACMode.OFF}]
        if hvac_mode in self._hvac_to_tuya:
            commands.append({'code': DPCode.MODE, 'value': self._hvac_to_tuya[hvac_mode]})
        self._send_command(commands)

    def set_preset_mode(self, preset_mode: str) -> None:
        """Set new target preset mode."""
        commands: List[Dict[str, Any]] = [{'code': DPCode.MODE, 'value': preset_mode}]
        self._send_command(commands)

    def set_fan_mode(self, fan_mode: str) -> None:
        """Set new target fan mode."""
        self._send_command([{'code': DPCode.FAN_SPEED_ENUM, 'value': fan_mode}])

    def set_humidity(self, humidity: int) -> None:
        """Set new target humidity."""
        if self._set_humidity is None:
            raise RuntimeError("Cannot set humidity, device doesn't provide methods to set it")
        self._send_command([
            {
                'code': self._set_humidity.dpcode,
                'value': self._set_humidity.scale_value_back(humidity),
            }
        ])

    def set_swing_mode(self, swing_mode: str) -> None:
        """Set new target swing operation."""
        self._send_command([
            {'code': DPCode.SHAKE, 'value': swing_mode == SWING_ON},
            {'code': DPCode.SWING, 'value': swing_mode == SWING_ON},
            {'code': DPCode.SWITCH_VERTICAL, 'value': swing_mode in (SWING_BOTH, SWING_VERTICAL)},
            {'code': DPCode.SWITCH_HORIZONTAL, 'value': swing_mode in (SWING_BOTH, SWING_HORIZONTAL)},
        ])

    def set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if self._set_temperature is None:
            raise RuntimeError("Cannot set target temperature, device doesn't provide methods to set it")
        temperature: float = kwargs.get('temperature')
        self._send_command([
            {
                'code': self._set_temperature.dpcode,
                'value': round(self._set_temperature.scale_value_back(temperature)),
            }
        ])

    @property
    def current_temperature(self) -> Optional[float]:
        """Return the current temperature."""
        if self._current_temperature is None:
            return None
        temperature = self.device.status.get(self._current_temperature.dpcode)
        if temperature is None:
            return None
        if self._current_temperature.scale == 0 and self._current_temperature.step != 1:
            temperature = temperature / 10
        return self._current_temperature.scale_value(temperature)

    @property
    def current_humidity(self) -> Optional[int]:
        """Return the current humidity."""
        if self._current_humidity is None:
            return None
        humidity = self.device.status.get(self._current_humidity.dpcode)
        if humidity is None:
            return None
        return round(self._current_humidity.scale_value(humidity))

    @property
    def target_temperature(self) -> Optional[float]:
        """Return the temperature currently set to be reached."""
        if self._set_temperature is None:
            return None
        temperature = self.device.status.get(self._set_temperature.dpcode)
        if temperature is None:
            return None
        return self._set_temperature.scale_value(temperature)

    @property
    def target_humidity(self) -> Optional[int]:
        """Return the humidity currently set to be reached."""
        if self._set_humidity is None:
            return None
        humidity = self.device.status.get(self._set_humidity.dpcode)
        if humidity is None:
            return None
        return round(self._set_humidity.scale_value(humidity))

    @property
    def hvac_mode(self) -> HVACMode:
        """Return hvac mode."""
        if not self.device.status.get(DPCode.SWITCH, True):
            return HVACMode.OFF
        if DPCode.MODE not in self.device.function:
            if self.device.status.get(DPCode.SWITCH, False):
                return self.entity_description.switch_only_hvac_mode
            return HVACMode.OFF
        mode = self.device.status.get(DPCode.MODE)
        if mode in TUYA_HVAC_TO_HA:
            return TUYA_HVAC_TO_HA[mode]
        if self.device.status.get(DPCode.SWITCH, False):
            return self.entity_description.switch_only_hvac_mode
        return HVACMode.OFF

    @property
    def preset_mode(self) -> Optional[str]:
        """Return preset mode."""
        if DPCode.MODE not in self.device.function:
            return None
        mode = self.device.status.get(DPCode.MODE)
        if mode in TUYA_HVAC_TO_HA:
            return None
        return mode

    @property
    def fan_mode(self) -> Optional[str]:
        """Return fan mode."""
        return self.device.status.get(DPCode.FAN_SPEED_ENUM)

    @property
    def swing_mode(self) -> str:
        """Return swing mode."""
        if any(
            (
                self.device.status.get(dpcode) is not None
                for dpcode in (DPCode.SHAKE, DPCode.SWING)
            )
        ):
            return SWING_ON
        horizontal = self.device.status.get(DPCode.SWITCH_HORIZONTAL)
        vertical = self.device.status.get(DPCode.SWITCH_VERTICAL)
        if horizontal and vertical:
            return SWING_BOTH
        if horizontal:
            return SWING_HORIZONTAL
        if vertical:
            return SWING_VERTICAL
        return SWING_OFF

    def turn_on(self) -> None:
        """Turn the device on, retaining current HVAC (if supported)."""
        self._send_command([{'code': DPCode.SWITCH, 'value': True}])

    def turn_off(self) -> None:
        """Turn the device off, retaining current HVAC (if supported)."""
        self._send_command([{'code': DPCode.SWITCH, 'value': False}])
