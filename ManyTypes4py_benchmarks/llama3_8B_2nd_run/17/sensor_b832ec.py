from __future__ import annotations
from collections.abc import Callable
from typing import Any
from homematicip.aio.device import AsyncBrandSwitchMeasuring, AsyncEnergySensorsInterface, AsyncFloorTerminalBlock6, AsyncFloorTerminalBlock10, AsyncFloorTerminalBlock12, AsyncFloorTerminalBlock12, AsyncFullFlushSwitchMeasuring, AsyncHeatingThermostat, AsyncHeatingThermostatCompact, AsyncHeatingThermostatEvo, AsyncHomeControlAccessPoint, AsyncLightSensor, AsyncMotionDetectorIndoor, AsyncMotionDetectorOutdoor, AsyncMotionDetectorPushButton, AsyncPassageDetector, AsyncPlugableSwitchMeasuring, AsyncPresenceDetectorIndoor, AsyncRoomControlDeviceAnalog, AsyncTemperatureDifferenceSensor2, AsyncTemperatureHumiditySensorDisplay, AsyncTemperatureHumiditySensorOutdoor, AsyncTemperatureHumiditySensorWithoutDisplay, AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro, AsyncWiredFloorTerminalBlock12
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
ILLUMINATION_DEVICE_ATTRIBUTES = {'currentIllumination': ATTR_CURRENT_ILLUMINATION, 'lowestIllumination': ATTR_LOWEST_ILLUMINATION, 'highestIllumination': ATTR_HIGHEST_ILLUMINATION}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: Callable[[list], None]) -> None:
    hap: HomematicipHAP = hass.data[DOMAIN][config_entry.unique_id]
    entities: list[HomematicipGenericEntity] = []
    for device in hap.home.devices:
        # ... rest of the code ...
        async_add_entities(entities)

class HomematicipFloorTerminalBlockMechanicChannelValve(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipAccesspointDutyCycle(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipHeatingThermostat(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipHumiditySensor(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipTemperatureSensor(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipIlluminanceSensor(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipPowerSensor(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipEnergySensor(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipWindspeedSensor(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicipTodayRainSensor(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicpTemperatureExternalSensorCh1(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicpTemperatureExternalSensorCh2(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HomematicpTemperatureExternalSensorDelta(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HmipEsiSensorEntity(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

class HmipEsiIecPowerConsumption(HmipEsiSensorEntity):
    # ... rest of the code ...

class HmipEsiIecEnergyCounterHighTariff(HmipEsiSensorEntity):
    # ... rest of the code ...

class HmipEsiIecEnergyCounterLowTariff(HmipEsiSensorEntity):
    # ... rest of the code ...

class HmipEsiIecEnergyCounterInputSingleTariff(HmipEsiSensorEntity):
    # ... rest of the code ...

class HmipEsiGasCurrentGasFlow(HmipEsiSensorEntity):
    # ... rest of the code ...

class HmipEsiGasGasVolume(HmipEsiSensorEntity):
    # ... rest of the code ...

class HmipEsiLedCurrentPowerConsumption(HmipEsiSensorEntity):
    # ... rest of the code ...

class HomematicipPassageDetectorDeltaCounter(HomematicipGenericEntity, SensorEntity):
    # ... rest of the code ...

def _get_wind_direction(wind_direction_degree: float) -> str:
    # ... rest of the code ...
