"""Support for Xiaomi Mi Air Quality Monitor (PM2.5) and Humidifier."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from miio import AirQualityMonitor, DeviceException
from miio.gateway.gateway import GATEWAY_MODEL_AC_V1, GATEWAY_MODEL_AC_V2, GATEWAY_MODEL_AC_V3, GATEWAY_MODEL_AQARA, GATEWAY_MODEL_EU, GatewayException
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_BATTERY_LEVEL, ATTR_TEMPERATURE, CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, CONCENTRATION_PARTS_PER_MILLION, CONF_DEVICE, CONF_HOST, CONF_MODEL, CONF_TOKEN, LIGHT_LUX, PERCENTAGE, REVOLUTIONS_PER_MINUTE, EntityCategory, UnitOfArea, UnitOfPower, UnitOfPressure, UnitOfTemperature, UnitOfTime, UnitOfVolume
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util
from . import VacuumCoordinatorDataAttributes
from .const import CONF_FLOW_TYPE, CONF_GATEWAY, DOMAIN, KEY_COORDINATOR, KEY_DEVICE, MODEL_AIRFRESH_A1, MODEL_AIRFRESH_T2017, MODEL_AIRFRESH_VA2, MODEL_AIRFRESH_VA4, MODEL_AIRHUMIDIFIER_CA1, MODEL_AIRHUMIDIFIER_CB1, MODEL_AIRPURIFIER_3C, MODEL_AIRPURIFIER_3C_REV_A, MODEL_AIRPURIFIER_4, MODEL_AIRPURIFIER_4_LITE_RMA1, MODEL_AIRPURIFIER_4_LITE_RMB1, MODEL_AIRPURIFIER_4_PRO, MODEL_AIRPURIFIER_MA2, MODEL_AIRPURIFIER_PRO, MODEL_AIRPURIFIER_PRO_V7, MODEL_AIRPURIFIER_V2, MODEL_AIRPURIFIER_V3, MODEL_AIRPURIFIER_ZA1, MODEL_FAN_P5, MODEL_FAN_V2, MODEL_FAN_V3, MODEL_FAN_ZA1, MODEL_FAN_ZA3, MODEL_FAN_ZA4, MODEL_FAN_ZA5, MODELS_AIR_QUALITY_MONITOR, MODELS_HUMIDIFIER_MIIO, MODELS_HUMIDIFIER_MIOT, MODELS_HUMIDIFIER_MJJSQ, MODELS_PURIFIER_MIIO, MODELS_PURIFIER_MIOT, MODELS_VACUUM, ROBOROCK_GENERIC, ROCKROBO_GENERIC
from .entity import XiaomiCoordinatedMiioEntity, XiaomiGatewayDevice, XiaomiMiioEntity
_LOGGER = logging.getLogger(__name__)
DEFAULT_NAME = 'Xiaomi Miio Sensor'
UNIT_LUMEN = 'lm'
ATTR_ACTUAL_SPEED = 'actual_speed'
ATTR_AIR_QUALITY = 'air_quality'
ATTR_TVOC = 'tvoc'
ATTR_AQI = 'aqi'
ATTR_BATTERY = 'battery'
ATTR_CARBON_DIOXIDE = 'co2'
ATTR_CHARGING = 'charging'
ATTR_CONTROL_SPEED = 'control_speed'
ATTR_DISPLAY_CLOCK = 'display_clock'
ATTR_FAVORITE_SPEED = 'favorite_speed'
ATTR_FILTER_LIFE_REMAINING = 'filter_life_remaining'
ATTR_FILTER_HOURS_USED = 'filter_hours_used'
ATTR_FILTER_LEFT_TIME = 'filter_left_time'
ATTR_DUST_FILTER_LIFE_REMAINING = 'dust_filter_life_remaining'
ATTR_DUST_FILTER_LIFE_REMAINING_DAYS = 'dust_filter_life_remaining_days'
ATTR_UPPER_FILTER_LIFE_REMAINING = 'upper_filter_life_remaining'
ATTR_UPPER_FILTER_LIFE_REMAINING_DAYS = 'upper_filter_life_remaining_days'
ATTR_FILTER_USE = 'filter_use'
ATTR_HUMIDITY = 'humidity'
ATTR_ILLUMINANCE = 'illuminance'
ATTR_ILLUMINANCE_LUX = 'illuminance_lux'
ATTR_LOAD_POWER = 'load_power'
ATTR_MOTOR2_SPEED = 'motor2_speed'
ATTR_MOTOR_SPEED = 'motor_speed'
ATTR_NIGHT_MODE = 'night_mode'
ATTR_NIGHT_TIME_BEGIN = 'night_time_begin'
ATTR_NIGHT_TIME_END = 'night_time_end'
ATTR_PM10 = 'pm10_density'
ATTR_PM25 = 'pm25'
ATTR_PM25_2 = 'pm25_2'
ATTR_POWER = 'power'
ATTR_PRESSURE = 'pressure'
ATTR_PURIFY_VOLUME = 'purify_volume'
ATTR_SENSOR_STATE = 'sensor_state'
ATTR_USE_TIME = 'use_time'
ATTR_WATER_LEVEL = 'water_level'
ATTR_DND_START = 'start'
ATTR_DND_END = 'end'
ATTR_LAST_CLEAN_TIME = 'duration'
ATTR_LAST_CLEAN_AREA = 'area'
ATTR_STATUS_CLEAN_TIME = 'clean_time'
ATTR_STATUS_CLEAN_AREA = 'clean_area'
ATTR_LAST_CLEAN_START = 'start'
ATTR_LAST_CLEAN_END = 'end'
ATTR_CLEAN_HISTORY_TOTAL_DURATION = 'total_duration'
ATTR_CLEAN_HISTORY_TOTAL_AREA = 'total_area'
ATTR_CLEAN_HISTORY_COUNT = 'count'
ATTR_CLEAN_HISTORY_DUST_COLLECTION_COUNT = 'dust_collection_count'
ATTR_CONSUMABLE_STATUS_MAIN_BRUSH_LEFT = 'main_brush_left'
ATTR_CONSUMABLE_STATUS_SIDE_BRUSH_LEFT = 'side_brush_left'
ATTR_CONSUMABLE_STATUS_FILTER_LEFT = 'filter_left'
ATTR_CONSUMABLE_STATUS_SENSOR_DIRTY_LEFT = 'sensor_dirty_left'

@dataclass(frozen=True)
class XiaomiMiioSensorDescription(SensorEntityDescription):
    """Class that holds device specific info for a xiaomi aqara or humidifier sensor."""
    attributes: Tuple[str, ...] = ()
    parent_key: Optional[str] = None

SENSOR_TYPES: Dict[str, XiaomiMiioSensorDescription] = {ATTR_TEMPERATURE: XiaomiMiioSensorDescription(key=ATTR_TEMPERATURE, native_unit_of_measurement=UnitOfTemperature.CELSIUS, device_class=SensorDeviceClass.TEMPERATURE, state_class=SensorStateClass.MEASUREMENT), ATTR_HUMIDITY: XiaomiMiioSensorDescription(key=ATTR_HUMIDITY, native_unit_of_measurement=PERCENTAGE, device_class=SensorDeviceClass.HUMIDITY, state_class=SensorStateClass.MEASUREMENT), ATTR_PRESSURE: XiaomiMiioSensorDescription(key=ATTR_PRESSURE, native_unit_of_measurement=UnitOfPressure.HPA, device_class=SensorDeviceClass.ATMOSPHERIC_PRESSURE, state_class=SensorStateClass.MEASUREMENT), ATTR_LOAD_POWER: XiaomiMiioSensorDescription(key=ATTR_LOAD_POWER, translation_key=ATTR_LOAD_POWER, native_unit_of_measurement=UnitOfPower.WATT, device_class=SensorDeviceClass.POWER), ATTR_WATER_LEVEL: XiaomiMiioSensorDescription(key=ATTR_WATER_LEVEL, translation_key=ATTR_WATER_LEVEL, native_unit_of_measurement=PERCENTAGE, icon='mdi:water-check', state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_ACTUAL_SPEED: XiaomiMiioSensorDescription(key=ATTR_ACTUAL_SPEED, translation_key=ATTR_ACTUAL_SPEED, native_unit_of_measurement=REVOLUTIONS_PER_MINUTE, icon='mdi:fast-forward', state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_CONTROL_SPEED: XiaomiMiioSensorDescription(key=ATTR_CONTROL_SPEED, translation_key=ATTR_CONTROL_SPEED, native_unit_of_measurement=REVOLUTIONS_PER_MINUTE, icon='mdi:fast-forward', state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_FAVORITE_SPEED: XiaomiMiioSensorDescription(key=ATTR_FAVORITE_SPEED, translation_key=ATTR_FAVORITE_SPEED, native_unit_of_measurement=REVOLUTIONS_PER_MINUTE, icon='mdi:fast-forward', state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_MOTOR_SPEED: XiaomiMiioSensorDescription(key=ATTR_MOTOR_SPEED, translation_key=ATTR_MOTOR_SPEED, native_unit_of_measurement=REVOLUTIONS_PER_MINUTE, icon='mdi:fast-forward', state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_MOTOR2_SPEED: XiaomiMiioSensorDescription(key=ATTR_MOTOR2_SPEED, translation_key=ATTR_MOTOR2_SPEED, native_unit_of_measurement=REVOLUTIONS_PER_MINUTE, icon='mdi:fast-forward', state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_USE_TIME: XiaomiMiioSensorDescription(key=ATTR_USE_TIME, translation_key=ATTR_USE_TIME, native_unit_of_measurement=UnitOfTime.SECONDS, icon='mdi:progress-clock', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.TOTAL_INCREASING, entity_registry_enabled_default=False, entity_category=EntityCategory.DIAGNOSTIC), ATTR_ILLUMINANCE: XiaomiMiioSensorDescription(key=ATTR_ILLUMINANCE, translation_key=ATTR_ILLUMINANCE, native_unit_of_measurement=UNIT_LUMEN, state_class=SensorStateClass.MEASUREMENT), ATTR_ILLUMINANCE_LUX: XiaomiMiioSensorDescription(key=ATTR_ILLUMINANCE, native_unit_of_measurement=LIGHT_LUX, device_class=SensorDeviceClass.ILLUMINANCE, state_class=SensorStateClass.MEASUREMENT), ATTR_AIR_QUALITY: XiaomiMiioSensorDescription(key=ATTR_AIR_QUALITY, translation_key=ATTR_AIR_QUALITY, native_unit_of_measurement='AQI', icon='mdi:cloud', state_class=SensorStateClass.MEASUREMENT), ATTR_TVOC: XiaomiMiioSensorDescription(key=ATTR_TVOC, translation_key=ATTR_TVOC, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, device_class=SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS), ATTR_PM10: XiaomiMiioSensorDescription(key=ATTR_PM10, native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, device_class=SensorDeviceClass.PM10, state_class=SensorStateClass.MEASUREMENT), ATTR_PM25: XiaomiMiioSensorDescription(key=ATTR_AQI, translation_key=ATTR_AQI, native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, device_class=SensorDeviceClass.PM25, state_class=SensorStateClass.MEASUREMENT), ATTR_PM25_2: XiaomiMiioSensorDescription(key=ATTR_PM25, native_unit_of_measurement=CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, device_class=SensorDeviceClass.PM25, state_class=SensorStateClass.MEASUREMENT), ATTR_FILTER_LIFE_REMAINING: XiaomiMiioSensorDescription(key=ATTR_FILTER_LIFE_REMAINING, translation_key=ATTR_FILTER_LIFE_REMAINING, native_unit_of_measurement=PERCENTAGE, icon='mdi:air-filter', state_class=SensorStateClass.MEASUREMENT, attributes=('filter_type',), entity_category=EntityCategory.DIAGNOSTIC), ATTR_FILTER_USE: XiaomiMiioSensorDescription(key=ATTR_FILTER_HOURS_USED, translation_key=ATTR_FILTER_HOURS_USED, native_unit_of_measurement=UnitOfTime.HOURS, icon='mdi:clock-outline', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_FILTER_LEFT_TIME: XiaomiMiioSensorDescription(key=ATTR_FILTER_LEFT_TIME, translation_key=ATTR_FILTER_LEFT_TIME, native_unit_of_measurement=UnitOfTime.DAYS, icon='mdi:clock-outline', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_DUST_FILTER_LIFE_REMAINING: XiaomiMiioSensorDescription(key=ATTR_DUST_FILTER_LIFE_REMAINING, translation_key=ATTR_DUST_FILTER_LIFE_REMAINING, native_unit_of_measurement=PERCENTAGE, icon='mdi:air-filter', state_class=SensorStateClass.MEASUREMENT, attributes=('filter_type',), entity_category=EntityCategory.DIAGNOSTIC), ATTR_DUST_FILTER_LIFE_REMAINING_DAYS: XiaomiMiioSensorDescription(key=ATTR_DUST_FILTER_LIFE_REMAINING_DAYS, translation_key=ATTR_DUST_FILTER_LIFE_REMAINING_DAYS, native_unit_of_measurement=UnitOfTime.DAYS, icon='mdi:clock-outline', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_UPPER_FILTER_LIFE_REMAINING: XiaomiMiioSensorDescription(key=ATTR_UPPER_FILTER_LIFE_REMAINING, translation_key=ATTR_UPPER_FILTER_LIFE_REMAINING, native_unit_of_measurement=PERCENTAGE, icon='mdi:air-filter', state_class=SensorStateClass.MEASUREMENT, attributes=('filter_type',), entity_category=EntityCategory.DIAGNOSTIC), ATTR_UPPER_FILTER_LIFE_REMAINING_DAYS: XiaomiMiioSensorDescription(key=ATTR_UPPER_FILTER_LIFE_REMAINING_DAYS, translation_key=ATTR_UPPER_FILTER_LIFE_REMAINING_DAYS, native_unit_of_measurement=UnitOfTime.DAYS, icon='mdi:clock-outline', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), ATTR_CARBON_DIOXIDE: XiaomiMiioSensorDescription(key=ATTR_CARBON_DIOXIDE, native_unit_of_measurement=CONCENTRATION_PARTS_PER_MILLION, device_class=SensorDeviceClass.CO2, state_class=SensorStateClass.MEASUREMENT), ATTR_PURIFY_VOLUME: XiaomiMiioSensorDescription(key=ATTR_PURIFY_VOLUME, translation_key=ATTR_PURIFY_VOLUME, native_unit_of_measurement=UnitOfVolume.CUBIC_METERS, device_class=SensorDeviceClass.VOLUME, state_class=SensorStateClass.TOTAL_INCREASING, entity_registry_enabled_default=False, entity_category=EntityCategory.DIAGNOSTIC), ATTR_BATTERY: XiaomiMiioSensorDescription(key=ATTR_BATTERY, native_unit_of_measurement=PERCENTAGE, device_class=SensorDeviceClass.BATTERY, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC)}
HUMIDIFIER_MIIO_SENSORS: Tuple[str, ...] = (ATTR_HUMIDITY, ATTR_TEMPERATURE, ATTR_USE_TIME, ATTR_WATER_LEVEL)
HUMIDIFIER_CA1_CB1_SENSORS: Tuple[str, ...] = (ATTR_HUMIDITY, ATTR_TEMPERATURE, ATTR_MOTOR_SPEED, ATTR_USE_TIME, ATTR_WATER_LEVEL)
HUMIDIFIER_MIOT_SENSORS: Tuple[str, ...] = (ATTR_ACTUAL_SPEED, ATTR_HUMIDITY, ATTR_TEMPERATURE, ATTR_USE_TIME, ATTR_WATER_LEVEL)
HUMIDIFIER_MJJSQ_SENSORS: Tuple[str, ...] = (ATTR_HUMIDITY, ATTR_TEMPERATURE)
PURIFIER_MIIO_SENSORS: Tuple[str, ...] = (ATTR_FILTER_LIFE_REMAINING, ATTR_FILTER_USE, ATTR_HUMIDITY, ATTR_MOTOR_SPEED, ATTR_PM25, ATTR_TEMPERATURE, ATTR