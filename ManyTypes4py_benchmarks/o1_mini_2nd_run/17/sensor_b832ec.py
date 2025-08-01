"""Support for HomematicIP Cloud sensors."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any, List, Optional
from homematicip.aio.device import (
    AsyncBrandSwitchMeasuring,
    AsyncEnergySensorsInterface,
    AsyncFloorTerminalBlock6,
    AsyncFloorTerminalBlock10,
    AsyncFloorTerminalBlock12,
    AsyncFullFlushSwitchMeasuring,
    AsyncHeatingThermostat,
    AsyncHeatingThermostatCompact,
    AsyncHeatingThermostatEvo,
    AsyncHomeControlAccessPoint,
    AsyncLightSensor,
    AsyncMotionDetectorIndoor,
    AsyncMotionDetectorOutdoor,
    AsyncMotionDetectorPushButton,
    AsyncPassageDetector,
    AsyncPlugableSwitchMeasuring,
    AsyncPresenceDetectorIndoor,
    AsyncRoomControlDeviceAnalog,
    AsyncTemperatureDifferenceSensor2,
    AsyncTemperatureHumiditySensorDisplay,
    AsyncTemperatureHumiditySensorOutdoor,
    AsyncTemperatureHumiditySensorWithoutDisplay,
    AsyncWeatherSensor,
    AsyncWeatherSensorPlus,
    AsyncWeatherSensorPro,
    AsyncWiredFloorTerminalBlock12,
)
from homematicip.base.enums import FunctionalChannelType, ValveState
from homematicip.base.functionalChannels import FloorTerminalBlockMechanicChannel, FunctionalChannel
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    LIGHT_LUX,
    PERCENTAGE,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfPrecipitationDepth,
    UnitOfSpeed,
    UnitOfTemperature,
    UnitOfVolume,
    UnitOfVolumeFlowRate,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from .const import DOMAIN
from .entity import HomematicipGenericEntity
from .hap import HomematicipHAP
from .helpers import get_channels_from_device

ATTR_CURRENT_ILLUMINATION: str = 'current_illumination'
ATTR_LOWEST_ILLUMINATION: str = 'lowest_illumination'
ATTR_HIGHEST_ILLUMINATION: str = 'highest_illumination'
ATTR_LEFT_COUNTER: str = 'left_counter'
ATTR_RIGHT_COUNTER: str = 'right_counter'
ATTR_TEMPERATURE_OFFSET: str = 'temperature_offset'
ATTR_WIND_DIRECTION: str = 'wind_direction'
ATTR_WIND_DIRECTION_VARIATION: str = 'wind_direction_variation_in_degree'
ATTR_ESI_TYPE: str = 'type'

ESI_TYPE_UNKNOWN: str = 'UNKNOWN'
ESI_CONNECTED_SENSOR_TYPE_IEC: str = 'ES_IEC'
ESI_CONNECTED_SENSOR_TYPE_GAS: str = 'ES_GAS'
ESI_CONNECTED_SENSOR_TYPE_LED: str = 'ES_LED'
ESI_TYPE_CURRENT_POWER_CONSUMPTION: str = 'CurrentPowerConsumption'
ESI_TYPE_ENERGY_COUNTER_USAGE_HIGH_TARIFF: str = 'ENERGY_COUNTER_USAGE_HIGH_TARIFF'
ESI_TYPE_ENERGY_COUNTER_USAGE_LOW_TARIFF: str = 'ENERGY_COUNTER_USAGE_LOW_TARIFF'
ESI_TYPE_ENERGY_COUNTER_INPUT_SINGLE_TARIFF: str = 'ENERGY_COUNTER_INPUT_SINGLE_TARIFF'
ESI_TYPE_CURRENT_GAS_FLOW: str = 'CurrentGasFlow'
ESI_TYPE_CURRENT_GAS_VOLUME: str = 'GasVolume'

ILLUMINATION_DEVICE_ATTRIBUTES: dict[str, str] = {
    'currentIllumination': ATTR_CURRENT_ILLUMINATION,
    'lowestIllumination': ATTR_LOWEST_ILLUMINATION,
    'highestIllumination': ATTR_HIGHEST_ILLUMINATION,
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the HomematicIP Cloud sensors from a config entry."""
    hap: HomematicipHAP = hass.data[DOMAIN][config_entry.unique_id]
    entities: List[SensorEntity] = []
    for device in hap.home.devices:
        if isinstance(device, AsyncHomeControlAccessPoint):
            entities.append(HomematicipAccesspointDutyCycle(hap, device))
        if isinstance(device, (AsyncHeatingThermostat, AsyncHeatingThermostatCompact, AsyncHeatingThermostatEvo)):
            entities.append(HomematicipHeatingThermostat(hap, device))
            entities.append(HomematicipTemperatureSensor(hap, device))
        if isinstance(device, (
            AsyncTemperatureHumiditySensorDisplay,
            AsyncTemperatureHumiditySensorWithoutDisplay,
            AsyncTemperatureHumiditySensorOutdoor,
            AsyncWeatherSensor,
            AsyncWeatherSensorPlus,
            AsyncWeatherSensorPro,
        )):
            entities.append(HomematicipTemperatureSensor(hap, device))
            entities.append(HomematicipHumiditySensor(hap, device))
        elif isinstance(device, (AsyncRoomControlDeviceAnalog,)):
            entities.append(HomematicipTemperatureSensor(hap, device))
        if isinstance(device, (
            AsyncLightSensor,
            AsyncMotionDetectorIndoor,
            AsyncMotionDetectorOutdoor,
            AsyncMotionDetectorPushButton,
            AsyncPresenceDetectorIndoor,
            AsyncWeatherSensor,
            AsyncWeatherSensorPlus,
            AsyncWeatherSensorPro,
        )):
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
        if isinstance(device, (
            AsyncFloorTerminalBlock6,
            AsyncFloorTerminalBlock10,
            AsyncFloorTerminalBlock12,
            AsyncWiredFloorTerminalBlock12,
        )):
            entities.extend((
                HomematicipFloorTerminalBlockMechanicChannelValve(hap, device, channel=channel.index)
                for channel in device.functionalChannels
                if isinstance(channel, FloorTerminalBlockMechanicChannel) and getattr(channel, 'valvePosition', None) is not None
            ))
    async_add_entities(entities)


class HomematicipFloorTerminalBlockMechanicChannelValve(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP floor terminal block."""

    _attr_native_unit_of_measurement: str = PERCENTAGE
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncFloorTerminalBlock6 | AsyncFloorTerminalBlock10 | AsyncFloorTerminalBlock12 | AsyncWiredFloorTerminalBlock12,
        channel: int,
        is_multi_channel: bool = True,
    ) -> None:
        """Initialize floor terminal block 12 device."""
        super().__init__(
            hap,
            device,
            channel=channel,
            is_multi_channel=is_multi_channel,
            post='Valve Position',
        )

    @property
    def icon(self) -> str | None:
        """Return the icon."""
        if super().icon:
            return super().icon
        channel: FloorTerminalBlockMechanicChannel = next(
            (channel for channel in self._device.functionalChannels if channel.index == self._channel),
            None,
        )
        if channel is None:
            return None
        if channel.valveState != ValveState.ADAPTION_DONE:
            return 'mdi:alert'
        return 'mdi:heating-coil'

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the floor terminal block mechanical channel valve position."""
        channel: FloorTerminalBlockMechanicChannel = next(
            (channel for channel in self._device.functionalChannels if channel.index == self._channel),
            None,
        )
        if channel is None or channel.valveState != ValveState.ADAPTION_DONE:
            return None
        return round(channel.valvePosition * 100)


class HomematicipAccesspointDutyCycle(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomeMaticIP access point."""

    _attr_icon: str = 'mdi:access-point-network'
    _attr_native_unit_of_measurement: str = PERCENTAGE
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncHomeControlAccessPoint) -> None:
        """Initialize access point status entity."""
        super().__init__(hap, device, post='Duty Cycle')

    @property
    def native_value(self) -> int:
        """Return the state of the access point."""
        return self._device.dutyCycleLevel


class HomematicipHeatingThermostat(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP heating thermostat."""

    _attr_native_unit_of_measurement: str = PERCENTAGE

    def __init__(self, hap: HomematicipHAP, device: AsyncHeatingThermostat | AsyncHeatingThermostatCompact | AsyncHeatingThermostatEvo) -> None:
        """Initialize heating thermostat device."""
        super().__init__(hap, device, post='Heating')

    @property
    def icon(self) -> str | None:
        """Return the icon."""
        if super().icon:
            return super().icon
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

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.HUMIDITY
    _attr_native_unit_of_measurement: str = PERCENTAGE
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncTemperatureHumiditySensorDisplay | AsyncTemperatureHumiditySensorWithoutDisplay | AsyncTemperatureHumiditySensorOutdoor | AsyncWeatherSensor | AsyncWeatherSensorPlus | AsyncWeatherSensorPro) -> None:
        """Initialize the humidity sensor device."""
        super().__init__(hap, device, post='Humidity')

    @property
    def native_value(self) -> Optional[float]:
        """Return the humidity state."""
        return self._device.humidity


class HomematicipTemperatureSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP temperature sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement: str = UnitOfTemperature.CELSIUS
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        """Initialize the temperature sensor device."""
        super().__init__(hap, device, post='Temperature')

    @property
    def native_value(self) -> Optional[float]:
        """Return the temperature state."""
        if hasattr(self._device, 'valveActualTemperature'):
            return self._device.valveActualTemperature
        return self._device.actualTemperature

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the temperature sensor."""
        state_attr: dict[str, Any] = super().extra_state_attributes
        temperature_offset: Optional[float] = getattr(self._device, 'temperatureOffset', None)
        if temperature_offset is not None:
            state_attr[ATTR_TEMPERATURE_OFFSET] = temperature_offset
        return state_attr


class HomematicipIlluminanceSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP Illuminance sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.ILLUMINANCE
    _attr_native_unit_of_measurement: str = LIGHT_LUX
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncLightSensor | AsyncMotionDetectorIndoor | AsyncMotionDetectorOutdoor | AsyncMotionDetectorPushButton | AsyncPresenceDetectorIndoor | AsyncWeatherSensor | AsyncWeatherSensorPlus | AsyncWeatherSensorPro) -> None:
        """Initialize the illuminance sensor device."""
        super().__init__(hap, device, post='Illuminance')

    @property
    def native_value(self) -> Optional[float]:
        """Return the illuminance state."""
        if hasattr(self._device, 'averageIllumination'):
            return self._device.averageIllumination
        return self._device.illumination

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the illuminance sensor."""
        state_attr: dict[str, Any] = super().extra_state_attributes
        for attr, attr_key in ILLUMINATION_DEVICE_ATTRIBUTES.items():
            attr_value = getattr(self._device, attr, None)
            if attr_value is not None:
                state_attr[attr_key] = attr_value
        return state_attr


class HomematicipPowerSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP power measuring sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement: str = UnitOfPower.WATT
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncPlugableSwitchMeasuring | AsyncBrandSwitchMeasuring | AsyncFullFlushSwitchMeasuring) -> None:
        """Initialize the power measuring sensor device."""
        super().__init__(hap, device, post='Power')

    @property
    def native_value(self) -> Optional[float]:
        """Return the power consumption value."""
        return self._device.currentPowerConsumption


class HomematicipEnergySensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP energy measuring sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement: str = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class: SensorStateClass = SensorStateClass.TOTAL_INCREASING

    def __init__(self, hap: HomematicipHAP, device: AsyncPlugableSwitchMeasuring | AsyncBrandSwitchMeasuring | AsyncFullFlushSwitchMeasuring) -> None:
        """Initialize the energy measuring sensor device."""
        super().__init__(hap, device, post='Energy')

    @property
    def native_value(self) -> Optional[float]:
        """Return the energy counter value."""
        return self._device.energyCounter


class HomematicipWindspeedSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP wind speed sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.WIND_SPEED
    _attr_native_unit_of_measurement: str = UnitOfSpeed.KILOMETERS_PER_HOUR
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncWeatherSensor | AsyncWeatherSensorPlus | AsyncWeatherSensorPro) -> None:
        """Initialize the windspeed sensor."""
        super().__init__(hap, device, post='Windspeed')

    @property
    def native_value(self) -> Optional[float]:
        """Return the wind speed value."""
        return self._device.windSpeed

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the wind speed sensor."""
        state_attr: dict[str, Any] = super().extra_state_attributes
        wind_direction: Optional[float] = getattr(self._device, 'windDirection', None)
        if wind_direction is not None:
            state_attr[ATTR_WIND_DIRECTION] = _get_wind_direction(wind_direction)
        wind_direction_variation: Optional[float] = getattr(self._device, 'windDirectionVariation', None)
        if wind_direction_variation is not None:
            state_attr[ATTR_WIND_DIRECTION_VARIATION] = wind_direction_variation
        return state_attr


class HomematicipTodayRainSensor(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP rain counter of a day sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.PRECIPITATION
    _attr_native_unit_of_measurement: str = UnitOfPrecipitationDepth.MILLIMETERS
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncWeatherSensorPlus | AsyncWeatherSensorPro) -> None:
        """Initialize the today rain sensor device."""
        super().__init__(hap, device, post='Today Rain')

    @property
    def native_value(self) -> Optional[float]:
        """Return today's rain value."""
        return round(self._device.todayRainCounter, 2)


class HomematicpTemperatureExternalSensorCh1(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP device HmIP-STE2-PCB."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement: str = UnitOfTemperature.CELSIUS
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncTemperatureDifferenceSensor2) -> None:
        """Initialize the external temperature channel 1 sensor device."""
        super().__init__(hap, device, post='Channel 1 Temperature')

    @property
    def native_value(self) -> Optional[float]:
        """Return the channel 1 temperature state."""
        return self._device.temperatureExternalOne


class HomematicpTemperatureExternalSensorCh2(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP device HmIP-STE2-PCB."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement: str = UnitOfTemperature.CELSIUS
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncTemperatureDifferenceSensor2) -> None:
        """Initialize the external temperature channel 2 sensor device."""
        super().__init__(hap, device, post='Channel 2 Temperature')

    @property
    def native_value(self) -> Optional[float]:
        """Return the channel 2 temperature state."""
        return self._device.temperatureExternalTwo


class HomematicpTemperatureExternalSensorDelta(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP device HmIP-STE2-PCB."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement: str = UnitOfTemperature.CELSIUS
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncTemperatureDifferenceSensor2) -> None:
        """Initialize the delta temperature sensor device."""
        super().__init__(hap, device, post='Delta Temperature')

    @property
    def native_value(self) -> Optional[float]:
        """Return the delta temperature state."""
        return self._device.temperatureExternalDelta


class HmipEsiSensorEntity(HomematicipGenericEntity, SensorEntity):
    """EntityDescription for HmIP-ESI Sensors."""

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncEnergySensorsInterface,
        key: str,
        value_fn: Callable[[FunctionalChannel], Any],
        type_fn: Callable[[FunctionalChannel], Any],
    ) -> None:
        """Initialize ESI sensor entity."""
        super().__init__(hap=hap, device=device, channel=1, post=key, is_multi_channel=False)
        self._value_fn: Callable[[FunctionalChannel], Any] = value_fn
        self._type_fn: Callable[[FunctionalChannel], Any] = type_fn

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the ESI sensor."""
        state_attr: dict[str, Any] = super().extra_state_attributes
        state_attr[ATTR_ESI_TYPE] = self._type_fn(self.functional_channel)
        return state_attr

    @property
    def native_value(self) -> Optional[str]:
        """Return the state of the ESI sensor."""
        return str(self._value_fn(self.functional_channel))


class HmipEsiIecPowerConsumption(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI IEC currentPowerConsumption sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement: str = UnitOfPower.WATT
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the IEC power consumption sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key='CurrentPowerConsumption',
            value_fn=lambda channel: channel.currentPowerConsumption,
            type_fn=lambda channel: 'CurrentPowerConsumption',
        )


class HmipEsiIecEnergyCounterHighTariff(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI IEC energyCounterOne sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement: str = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class: SensorStateClass = SensorStateClass.TOTAL_INCREASING

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the IEC energy counter high tariff sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key=ESI_TYPE_ENERGY_COUNTER_USAGE_HIGH_TARIFF,
            value_fn=lambda channel: channel.energyCounterOne,
            type_fn=lambda channel: channel.energyCounterOneType,
        )


class HmipEsiIecEnergyCounterLowTariff(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI IEC energyCounterTwo sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement: str = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class: SensorStateClass = SensorStateClass.TOTAL_INCREASING

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the IEC energy counter low tariff sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key=ESI_TYPE_ENERGY_COUNTER_USAGE_LOW_TARIFF,
            value_fn=lambda channel: channel.energyCounterTwo,
            type_fn=lambda channel: channel.energyCounterTwoType,
        )


class HmipEsiIecEnergyCounterInputSingleTariff(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI IEC energyCounterThree sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement: str = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class: SensorStateClass = SensorStateClass.TOTAL_INCREASING

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the IEC energy counter input single tariff sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key=ESI_TYPE_ENERGY_COUNTER_INPUT_SINGLE_TARIFF,
            value_fn=lambda channel: channel.energyCounterThree,
            type_fn=lambda channel: channel.energyCounterThreeType,
        )


class HmipEsiGasCurrentGasFlow(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI Gas currentGasFlow sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.VOLUME_FLOW_RATE
    _attr_native_unit_of_measurement: str = UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the Gas currentGasFlow sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key='CurrentGasFlow',
            value_fn=lambda channel: channel.currentGasFlow,
            type_fn=lambda channel: 'CurrentGasFlow',
        )


class HmipEsiGasGasVolume(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI Gas gasVolume sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.GAS
    _attr_native_unit_of_measurement: str = UnitOfVolume.CUBIC_METERS
    _attr_state_class: SensorStateClass = SensorStateClass.TOTAL_INCREASING

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the Gas gasVolume sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key='GasVolume',
            value_fn=lambda channel: channel.gasVolume,
            type_fn=lambda channel: 'GasVolume',
        )


class HmipEsiLedCurrentPowerConsumption(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI LED currentPowerConsumption sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement: str = UnitOfPower.WATT
    _attr_state_class: SensorStateClass = SensorStateClass.MEASUREMENT

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the LED currentPowerConsumption sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key='CurrentPowerConsumption',
            value_fn=lambda channel: channel.currentPowerConsumption,
            type_fn=lambda channel: 'CurrentPowerConsumption',
        )


class HmipEsiLedEnergyCounterHighTariff(HmipEsiSensorEntity):
    """Representation of the Hmip-ESI LED energyCounterOne sensor."""

    _attr_device_class: SensorDeviceClass = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement: str = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class: SensorStateClass = SensorStateClass.TOTAL_INCREASING

    def __init__(self, hap: HomematicipHAP, device: AsyncEnergySensorsInterface) -> None:
        """Initialize the LED energy counter high tariff sensor."""
        super().__init__(
            hap=hap,
            device=device,
            key=ESI_TYPE_ENERGY_COUNTER_USAGE_HIGH_TARIFF,
            value_fn=lambda channel: channel.energyCounterOne,
            type_fn=lambda channel: ESI_TYPE_ENERGY_COUNTER_USAGE_HIGH_TARIFF,
        )


class HomematicipPassageDetectorDeltaCounter(HomematicipGenericEntity, SensorEntity):
    """Representation of the HomematicIP passage detector delta counter."""

    def __init__(self, hap: HomematicipHAP, device: AsyncPassageDetector) -> None:
        """Initialize the passage detector delta counter entity."""
        super().__init__(hap, device, post='Passage Detector Delta Counter')

    @property
    def native_value(self) -> Optional[int]:
        """Return the passage detector delta counter value."""
        return self._device.leftRightCounterDelta

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the delta counter."""
        state_attr: dict[str, Any] = super().extra_state_attributes
        state_attr[ATTR_LEFT_COUNTER] = self._device.leftCounter
        state_attr[ATTR_RIGHT_COUNTER] = self._device.rightCounter
        return state_attr


def _get_wind_direction(wind_direction_degree: float) -> str:
    """Convert wind direction degree to named direction."""
    if 11.25 <= wind_direction_degree < 33.75:
        return 'NNE'
    if 33.75 <= wind_direction_degree < 56.25:
        return 'NE'
    if 56.25 <= wind_direction_degree < 78.75:
        return 'ENE'
    if 78.75 <= wind_direction_degree < 101.25:
        return 'E'
    if 101.25 <= wind_direction_degree < 123.75:
        return 'ESE'
    if 123.75 <= wind_direction_degree < 146.25:
        return 'SE'
    if 146.25 <= wind_direction_degree < 168.75:
        return 'SSE'
    if 168.75 <= wind_direction_degree < 191.25:
        return 'S'
    if 191.25 <= wind_direction_degree < 213.75:
        return 'SSW'
    if 213.75 <= wind_direction_degree < 236.25:
        return 'SW'
    if 236.25 <= wind_direction_degree < 258.75:
        return 'WSW'
    if 258.75 <= wind_direction_degree < 281.25:
        return 'W'
    if 281.25 <= wind_direction_degree < 303.75:
        return 'WNW'
    if 303.75 <= wind_direction_degree < 326.25:
        return 'NW'
    if 326.25 <= wind_direction_degree < 348.75:
        return 'NNW'
    return 'N'
