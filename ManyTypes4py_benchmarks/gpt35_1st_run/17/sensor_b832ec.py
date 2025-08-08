from __future__ import annotations
from collections.abc import Callable
from typing import Any, List
from homematicip.aio.device import AsyncBrandSwitchMeasuring, AsyncEnergySensorsInterface, AsyncFloorTerminalBlock6, AsyncFloorTerminalBlock10, AsyncFloorTerminalBlock12, AsyncFullFlushSwitchMeasuring, AsyncHeatingThermostat, AsyncHeatingThermostatCompact, AsyncHeatingThermostatEvo, AsyncHomeControlAccessPoint, AsyncLightSensor, AsyncMotionDetectorIndoor, AsyncMotionDetectorOutdoor, AsyncMotionDetectorPushButton, AsyncPassageDetector, AsyncPlugableSwitchMeasuring, AsyncPresenceDetectorIndoor, AsyncRoomControlDeviceAnalog, AsyncTemperatureDifferenceSensor2, AsyncTemperatureHumiditySensorDisplay, AsyncTemperatureHumiditySensorOutdoor, AsyncTemperatureHumiditySensorWithoutDisplay, AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro, AsyncWiredFloorTerminalBlock12
from homematicip.base.enums import FunctionalChannelType, ValveState
from homematicip.base.functionalChannels import FloorTerminalBlockMechanicChannel, FunctionalChannel
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import LIGHT_LUX, PERCENTAGE, UnitOfEnergy, UnitOfPower, UnitOfPrecipitationDepth, UnitOfSpeed, UnitOfTemperature, UnitOfVolume, UnitOfVolumeFlowRate
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
ILLUMINATION_DEVICE_ATTRIBUTES: dict = {'currentIllumination': ATTR_CURRENT_ILLUMINATION, 'lowestIllumination': ATTR_LOWEST_ILLUMINATION, 'highestIllumination': ATTR_HIGHEST_ILLUMINATION}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the HomematicIP Cloud sensors from a config entry."""
    hap: HomematicipHAP = hass.data[DOMAIN][config_entry.unique_id]
    entities: List[HomematicipGenericEntity] = []
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
