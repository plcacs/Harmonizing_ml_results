"""Support for HomematicIP Cloud sensors."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any, Optional, cast
from homematicip.aio.device import AsyncBrandSwitchMeasuring, AsyncEnergySensorsInterface, AsyncFloorTerminalBlock6, AsyncFloorTerminalBlock10, AsyncFloorTerminalBlock12, AsyncFullFlushSwitchMeasuring, AsyncHeatingThermostat, AsyncHeatingThermostatCompact, AsyncHeatingThermostatEvo, AsyncHomeControlAccessPoint, AsyncLightSensor, AsyncMotionDetectorIndoor, AsyncMotionDetectorOutdoor, AsyncMotionDetectorPushButton, AsyncPassageDetector, AsyncPlugableSwitchMeasuring, AsyncPresenceDetectorIndoor, AsyncRoomControlDeviceAnalog, AsyncTemperatureDifferenceSensor2, AsyncTemperatureHumiditySensorDisplay, AsyncTemperatureHumiditySensorOutdoor, AsyncTemperatureHumiditySensorWithoutDisplay, AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro, AsyncWiredFloorTerminalBlock12
from homematicip.base.enums import FunctionalChannelType, ValveState
from homematicip.base.functionalChannels import FloorTerminalBlockMechanicChannel, FunctionalChannel
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import LIGHT_LUX, PERCENTAGE, UnitOfEnergy, UnitOfPower, UnitOfPrecipitationDepth, UnitOfSpeed, UnitOfTemperature, UnitOfVolume, UnitOfVolumeFlowRate
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType
from .const import DOMAIN
from .entity import HomematicipGenericEntity
from .hap import HomematicipHAP
from .helpers import get_channels_from_device
from typing import Dict, List

ATTR_CURRENT_ILLUMINATION = 'current_illumination'
ATTR_LOWEST_ILLUMINATION = 'lowest_illumination'
ATTR_HIGHEST_ILLUMINATION = 'highest_illumination'
ATTR_LEFT_COUNTER = 'left_counter'
ATTR_RIGHT_COUNTER = 'right_counter'
ATTR_TEMPERATURE_OFFSET = 'temperature_offset'
ATTR_WIND_DIRECTION = 'wind_direction'
ATTR_WIND_DIRECTION_VARIATION = 'wind_direction_variation_in_degree'
ATTR_ESI_TYPE = 'type'
ESI_TYPE_UNKNOWN = 'UNKNOWN'
ESI_CONNECTED_SENSOR_TYPE_IEC = 'ES_IEC'
ESI_CONNECTED_SENSOR_TYPE_GAS = 'ES_GAS'
ESI_CONNECTED_SENSOR_TYPE_LED = 'ES_LED'
ESI_TYPE_CURRENT_POWER_CONSUMPTION = 'CurrentPowerConsumption'
ESI_TYPE_ENERGY_COUNTER_USAGE_HIGH_TARIFF = 'ENERGY_COUNTER_USAGE_HIGH_TARIFF'
ESI_TYPE_ENERGY_COUNTER_USAGE_LOW_TARIFF = 'ENERGY_COUNTER_USAGE_LOW_TARIFF'
ESI_TYPE_ENERGY_COUNTER_INPUT_SINGLE_TARIFF = 'ENERGY_COUNTER_INPUT_SINGLE_TARIFF'
ESI_TYPE_CURRENT_GAS_FLOW = 'CurrentGasFlow'
ESI_TYPE_CURRENT_GAS_VOLUME = 'GasVolume'
ILLUMINATION_DEVICE_ATTRIBUTES: Dict[str, str] = {'currentIllumination': ATTR_CURRENT_ILLUMINATION, 'lowestIllumination': ATTR_LOWEST_ILLUMINATION, 'highestIllumination': ATTR_HIGHEST_ILLUMINATION}

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the HomematicIP Cloud sensors from a config entry."""
    hap = hass.data[DOMAIN][config_entry.unique_id]
    entities: List[SensorEntity] = []
    for device in hap.home.devices:
        if isinstance(device, AsyncHomeControlAccessPoint):
            entities.append(HomematicipAccesspointDutyCycle(hap, device))
        if isinstance(device, (AsyncHeatingThermostat, AsyncHeatingThermostatCompact, AsyncHeatingThermostatEvo)):
            entities.append(HomematicipHeatingThermostat(hap, device))
            entities.append(HomematicipTemperatureSensor(hap, device))
        if isinstance(device, (AsyncTemperatureHumiditySensorDisplay, AsyncTemperatureHumiditySensorWithoutDisplay, AsyncTemperatureHumiditySensorOutdoor, AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro)):
            entities.append(HomematicipTemperatureSensor(hap, device))
            entities.append(HomematicipHumiditySensor(hap, device))
        elif isinstance(device, (AsyncRoomControlDeviceAnalog,)):
            entities.append(HomematicipTemperatureSensor(hap, device))
        if isinstance(device, (AsyncLightSensor, AsyncMotionDetectorIndoor, AsyncMotionDetectorOutdoor, AsyncMotionDetectorPushButton, AsyncPresenceDetectorIndoor, AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro)):
            entities.append(HomematicipIlluminanceSensor(hap, device))
        if isinstance(device, (AsyncPlugableSwitchMeasuring, AsyncBrandSwitchMeasuring, AsyncFullFlushSwitchMeasuring)):
            entities.append(HomematicipPowerSensor(hap, device))
            entities.append(HomematicipEnergySensor(hap, device))
        if isinstance(device, (AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro)):
            entities.append(HomematicipWindspeedSensor(hap, device))
        if isinstance(device, (AsyncWeatherSensorPlus, AsyncWeatherSensorPro)):
            entities.append(HomematicipTodayRainSensor(hap, device))
        if isinstance(device, AsyncPassageDetector):
            entities.append(HomematicipPassageDetectorDeltaCounter(hap, device))
        if isinstance(device, AsyncTemperatureDifferenceSensor2):
            entities.append(HomematicpTemperatureExternalSensorCh1(hap, device))
            entities.append(HomematicpTemperatureExternalSensorCh2(hap, device))
            entities.append(HomematicpTemperatureExternalSensorDelta(hap, device))
        if isinstance(device, AsyncEnergySensorsInterface):
            for ch in get_channels_from_device(device, FunctionalChannelType.ENERGY_SENSORS_INTERFACE_CHANNEL):
                if ch.connectedEnergySensorType == ESI_CONNECTED_SENSOR_TYPE_IEC:
                    if ch.currentPowerConsumption is not None:
                        entities.append(HmipEsiIecPowerConsumption(hap, device))
                    if ch.energyCounterOneType != ESI_TYPE_UNKNOWN:
                        entities.append(HmipEsiIecEnergyCounterHighTariff(hap, device))
                    if ch.energyCounterTwoType != ESI_TYPE_UNKNOWN:
                        entities.append(HmipEsiIecEnergyCounterLowTariff(hap, device))
                    if ch.energyCounterThreeType != ESI_TYPE_UNKNOWN:
                        entities.append(HmipEsiIecEnergyCounterInputSingleTariff(hap, device))
                if ch.connectedEnergySensorType == ESI_CONNECTED_SENSOR_TYPE_GAS:
                    if ch.currentGasFlow is not None:
                        entities.append(HmipEsiGasCurrentGasFlow(hap, device))
                    if ch.gasVolume is not None:
                        entities.append(HmipEsiGasGasVolume(hap, device))
                if ch.connectedEnergySensorType == ESI_CONNECTED_SENSOR_TYPE_LED:
                    if ch.currentPowerConsumption is not None:
                        entities.append(HmipEsiLedCurrentPowerConsumption(hap, device))
                    entities.append(HmipEsiLedEnergyCounterHighTariff(hap, device))
        if isinstance(device, (AsyncFloorTerminalBlock6, AsyncFloorTerminalBlock10, AsyncFloorTerminalBlock12, AsyncWiredFloorTerminalBlock12)):
            entities.extend((HomematicipFloorTerminalBlockMechanicChannelValve(hap, device, channel=channel.index) for channel in device.functionalChannels if isinstance(channel, FloorTerminalBlockMechanicChannel) and getattr(channel, 'valvePosition', None) is not None))
    async_add_entities(entities)

class HomematicipFloorTerminalBlockMechanicChannelValve(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP floor terminal block."""
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any, channel: int, is_multi_channel: bool = True) -> None:
        """Initialize floor terminal block 12 device."""
        super().__init__(hap, device, channel=channel, is_multi_channel=is_multi_channel, post='Valve Position')

    @property
    def icon(self) -> str:
        """Return the icon."""
        if super().icon:
            return cast(str, super().icon)
        channel = next((channel for channel in self._device.functionalChannels if channel.index == self._channel))
        if channel.valveState != ValveState.ADAPTION_DONE:
            return 'mdi:alert'
        return 'mdi:heating-coil'

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the floor terminal block mechanical channel valve position."""
        channel = next((channel for channel in self._device.functionalChannels if channel.index == self._channel))
        if channel.valveState != ValveState.ADAPTION_DONE:
            return None
        return round(channel.valvePosition * 100)

class HomematicipAccesspointDutyCycle(HomematicipGenericEntity, SensorEntity):
    """Representation of then HomeMaticIP access point."""
    _attr_icon = 'mdi:access-point-network'
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize access point status entity."""
        super().__init__(hap, device, post='Duty Cycle')

    @property
    def native_value(self) -> float:
        """Return the state of the access point."""
        return self._device.dutyCycleLevel

class HomematicipHeatingThermostat(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP heating thermostat."""
    _attr_native_unit_of_measurement = PERCENTAGE

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize heating thermostat device."""
        super().__init__(hap, device, post='Heating')

    @property
    def icon(self) -> str:
        """Return the icon."""
        if super().icon:
            return cast(str, super().icon)
        if self._device.valveState != ValveState.ADAPTION_DONE:
            return 'mdi:alert'
        return 'mdi:radiator'

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the radiator valve."""
        if self._device.valveState != ValveState.ADAPTION_DONE:
            return None
        return round(self._device.valvePosition * 100)

class HomematicipHumiditySensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP humidity sensor."""
    _attr_device_class = SensorDeviceClass.HUMIDITY
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the thermometer device."""
        super().__init__(hap, device, post='Humidity')

    @property
    def native_value(self) -> float:
        """Return the state."""
        return self._device.humidity

class HomematicipTemperatureSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP thermometer."""
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the thermometer device."""
        super().__init__(hap, device, post='Temperature')

    @property
    def native_value(self) -> float:
        """Return the state."""
        if hasattr(self._device, 'valveActualTemperature'):
            return self._device.valveActualTemperature
        return self._device.actualTemperature

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the windspeed sensor."""
        state_attr = super().extra_state_attributes
        temperature_offset = getattr(self._device, 'temperatureOffset', None)
        if temperature_offset:
            state_attr[ATTR_TEMPERATURE_OFFSET] = temperature_offset
        return state_attr

class HomematicipIlluminanceSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP Illuminance sensor."""
    _attr_device_class = SensorDeviceClass.ILLUMINANCE
    _attr_native_unit_of_measurement = LIGHT_LUX
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the  device."""
        super().__init__(hap, device, post='Illuminance')

    @property
    def native_value(self) -> float:
        """Return the state."""
        if hasattr(self._device, 'averageIllumination'):
            return self._device.averageIllumination
        return self._device.illumination

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the wind speed sensor."""
        state_attr = super().extra_state_attributes
        for attr, attr_key in ILLUMINATION_DEVICE_ATTRIBUTES.items():
            if (attr_value := getattr(self._device, attr, None)):
                state_attr[attr_key] = attr_value
        return state_attr

class HomematicipPowerSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP power measuring sensor."""
    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the  device."""
        super().__init__(hap, device, post='Power')

    @property
    def native_value(self) -> float:
        """Return the power consumption value."""
        return self._device.currentPowerConsumption

class HomematicipEnergySensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP energy measuring sensor."""
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL_INCREASING

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the device."""
        super().__init__(hap, device, post='Energy')

    @property
    def native_value(self) -> float:
        """Return the energy counter value."""
        return self._device.energyCounter

class HomematicipWindspeedSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP wind speed sensor."""
    _attr_device_class = SensorDeviceClass.WIND_SPEED
    _attr_native_unit_of_measurement = UnitOfSpeed.KILOMETERS_PER_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the windspeed sensor."""
        super().__init__(hap, device, post='Windspeed')

    @property
    def native_value(self) -> float:
        """Return the wind speed value."""
        return self._device.windSpeed

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the wind speed sensor."""
        state_attr = super().extra_state_attributes
        wind_direction = getattr(self._device, 'windDirection', None)
        if wind_direction is not None:
            state_attr[ATTR_WIND_DIRECTION] = _get_wind_direction(wind_direction)
        wind_direction_variation = getattr(self._device, 'windDirectionVariation', None)
        if wind_direction_variation:
            state_attr[ATTR_WIND_DIRECTION_VARIATION] = wind_direction_variation
        return state_attr

class HomematicipTodayRainSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP rain counter of a day sensor."""
    _attr_device_class = SensorDeviceClass.PRECIPITATION
    _attr_native_unit_of_measurement = UnitOfPrecipitationDepth.MILLIMETERS
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the  device."""
        super().__init__(hap, device, post='Today Rain')

    @property
    def native_value(self) -> float:
        """Return the today's rain value."""
        return round(self._device.todayRainCounter, 2)

class HomematicpTemperatureExternalSensorCh1(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP device HmIP-STE2-PCB."""
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the  device."""
        super().__init__(hap, device,