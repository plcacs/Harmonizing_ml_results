"""Support for the Fitbit API."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import datetime
import logging
from typing import Any, Final, cast
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfLength, UnitOfMass, UnitOfTime, UnitOfVolume
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.icon import icon_for_battery_level
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .api import FitbitApi
from .const import ATTRIBUTION, BATTERY_LEVELS, DOMAIN, FitbitScope, FitbitUnitSystem
from .coordinator import FitbitConfigEntry, FitbitDeviceCoordinator
from .exceptions import FitbitApiException, FitbitAuthException
from .model import FitbitDevice, config_from_entry_data
_LOGGER: Final = logging.getLogger(__name__)
_CONFIGURING: dict[str, Any] = {}
SCAN_INTERVAL: Final = datetime.timedelta(minutes=30)
FITBIT_TRACKER_SUBSTRING: Final = '/tracker/'

def _default_value_fn(result: dict[str, Any]) -> str:
    """Parse a Fitbit timeseries API responses."""
    return cast(str, result['value'])

def _distance_value_fn(result: dict[str, Any]) -> str:
    """Format function for distance values."""
    return format(float(_default_value_fn(result)), '.2f')

def _body_value_fn(result: dict[str, Any]) -> str:
    """Format function for body values."""
    return format(float(_default_value_fn(result)), '.1f')

def _clock_format_12h(result: dict[str, Any]) -> str:
    raw_state = result['value']
    if raw_state == '':
        return '-'
    (hours_str, minutes_str) = raw_state.split(':')
    (hours, minutes) = (int(hours_str), int(minutes_str))
    setting = 'AM'
    if hours > 12:
        setting = 'PM'
        hours -= 12
    elif hours == 0:
        hours = 12
    return f'{hours}:{minutes:02d} {setting}'

def _weight_unit(unit_system: FitbitUnitSystem) -> str:
    """Determine the weight unit."""
    if unit_system == FitbitUnitSystem.EN_US:
        return UnitOfMass.POUNDS
    if unit_system == FitbitUnitSystem.EN_GB:
        return UnitOfMass.STONES
    return UnitOfMass.KILOGRAMS

def _distance_unit(unit_system: FitbitUnitSystem) -> str:
    """Determine the distance unit."""
    if unit_system == FitbitUnitSystem.EN_US:
        return UnitOfLength.MILES
    return UnitOfLength.KILOMETERS

def _elevation_unit(unit_system: FitbitUnitSystem) -> str:
    """Determine the elevation unit."""
    if unit_system == FitbitUnitSystem.EN_US:
        return UnitOfLength.FEET
    return UnitOfLength.METERS

def _water_unit(unit_system: FitbitUnitSystem) -> str:
    """Determine the water unit."""
    if unit_system == FitbitUnitSystem.EN_US:
        return UnitOfVolume.FLUID_OUNCES
    return UnitOfVolume.MILLILITERS

def _int_value_or_none(field: str) -> Callable[[dict[str, Any]], int | None]:
    """Value function that will parse the specified field if present."""

    def convert(result: dict[str, Any]) -> int | None:
        if (value := result['value'].get(field)) is not None:
            return int(value)
        return None
    return convert

@dataclass(frozen=True)
class FitbitSensorEntityDescription(SensorEntityDescription):
    """Describes Fitbit sensor entity."""
    unit_type: str | None = None
    value_fn: Callable[[dict[str, Any]], Any] = _default_value_fn
    unit_fn: Callable[[FitbitUnitSystem], str | None] = lambda x: None
    scope: FitbitScope | None = None

    @property
    def is_tracker(self) -> bool:
        """Return if the entity is a tracker."""
        return FITBIT_TRACKER_SUBSTRING in self.key

def _build_device_info(config_entry: FitbitConfigEntry, entity_description: FitbitSensorEntityDescription) -> DeviceInfo:
    """Build device info for sensor entities info across devices."""
    unique_id = cast(str, config_entry.unique_id)
    if entity_description.is_tracker:
        return DeviceInfo(entry_type=DeviceEntryType.SERVICE, identifiers={(DOMAIN, f'{unique_id}_tracker')}, translation_key='tracker', translation_placeholders={'display_name': config_entry.title})
    return DeviceInfo(entry_type=DeviceEntryType.SERVICE, identifiers={(DOMAIN, unique_id)})
FITBIT_RESOURCES_LIST: tuple[FitbitSensorEntityDescription, ...] = (FitbitSensorEntityDescription(key='activities/activityCalories', translation_key='activity_calories', native_unit_of_measurement='cal', icon='mdi:fire', scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/calories', translation_key='calories', native_unit_of_measurement='cal', icon='mdi:fire', scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.TOTAL_INCREASING), FitbitSensorEntityDescription(key='activities/caloriesBMR', translation_key='calories_bmr', native_unit_of_measurement='cal', icon='mdi:fire', scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/distance', icon='mdi:map-marker', device_class=SensorDeviceClass.DISTANCE, value_fn=_distance_value_fn, unit_fn=_distance_unit, scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.TOTAL_INCREASING), FitbitSensorEntityDescription(key='activities/elevation', translation_key='elevation', icon='mdi:walk', device_class=SensorDeviceClass.DISTANCE, unit_fn=_elevation_unit, scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/floors', translation_key='floors', native_unit_of_measurement='floors', icon='mdi:walk', scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/heart', translation_key='resting_heart_rate', native_unit_of_measurement='bpm', icon='mdi:heart-pulse', value_fn=_int_value_or_none('restingHeartRate'), scope=FitbitScope.HEART_RATE, state_class=SensorStateClass.MEASUREMENT), FitbitSensorEntityDescription(key='activities/minutesFairlyActive', translation_key='minutes_fairly_active', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:walk', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/minutesLightlyActive', translation_key='minutes_lightly_active', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:walk', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/minutesSedentary', translation_key='minutes_sedentary', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:seat-recline-normal', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/minutesVeryActive', translation_key='minutes_very_active', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:run', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/steps', translation_key='steps', native_unit_of_measurement='steps', icon='mdi:walk', scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.TOTAL_INCREASING), FitbitSensorEntityDescription(key='activities/tracker/activityCalories', translation_key='activity_calories', native_unit_of_measurement='cal', icon='mdi:fire', scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/calories', translation_key='calories', native_unit_of_measurement='cal', icon='mdi:fire', scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/distance', icon='mdi:map-marker', device_class=SensorDeviceClass.DISTANCE, value_fn=_distance_value_fn, unit_fn=_distance_unit, scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/elevation', translation_key='elevation', icon='mdi:walk', device_class=SensorDeviceClass.DISTANCE, unit_fn=_elevation_unit, scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/floors', translation_key='floors', native_unit_of_measurement='floors', icon='mdi:walk', scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/minutesFairlyActive', translation_key='minutes_fairly_active', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:walk', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/minutesLightlyActive', translation_key='minutes_lightly_active', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:walk', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/minutesSedentary', translation_key='minutes_sedentary', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:seat-recline-normal', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/minutesVeryActive', translation_key='minutes_very_active', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:run', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='activities/tracker/steps', translation_key='steps', native_unit_of_measurement='steps', icon='mdi:walk', scope=FitbitScope.ACTIVITY, entity_registry_enabled_default=False, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='body/bmi', translation_key='bmi', native_unit_of_measurement='BMI', icon='mdi:human', state_class=SensorStateClass.MEASUREMENT, value_fn=_body_value_fn, scope=FitbitScope.WEIGHT, entity_registry_enabled_default=False, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='body/fat', translation_key='body_fat', native_unit_of_measurement=PERCENTAGE, icon='mdi:human', state_class=SensorStateClass.MEASUREMENT, value_fn=_body_value_fn, scope=FitbitScope.WEIGHT, entity_registry_enabled_default=False, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='body/weight', icon='mdi:human', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.WEIGHT, value_fn=_body_value_fn, unit_fn=_weight_unit, scope=FitbitScope.WEIGHT), FitbitSensorEntityDescription(key='sleep/awakeningsCount', translation_key='awakenings_count', native_unit_of_measurement='times awaken', icon='mdi:sleep', scope=FitbitScope.SLEEP, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='sleep/efficiency', translation_key='sleep_efficiency', native_unit_of_measurement=PERCENTAGE, icon='mdi:sleep', state_class=SensorStateClass.MEASUREMENT, scope=FitbitScope.SLEEP, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='sleep/minutesAfterWakeup', translation_key='minutes_after_wakeup', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:sleep', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.SLEEP, state_class=SensorStateClass.MEASUREMENT, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='sleep/minutesAsleep', translation_key='sleep_minutes_asleep', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:sleep', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.SLEEP, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='sleep/minutesAwake', translation_key='sleep_minutes_awake', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:sleep', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.SLEEP, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='sleep/minutesToFallAsleep', translation_key='sleep_minutes_to_fall_asleep', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:sleep', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.SLEEP, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='sleep/timeInBed', translation_key='sleep_time_in_bed', native_unit_of_measurement=UnitOfTime.MINUTES, icon='mdi:hotel', device_class=SensorDeviceClass.DURATION, scope=FitbitScope.SLEEP, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='foods/log/caloriesIn', translation_key='calories_in', native_unit_of_measurement='cal', icon='mdi:food-apple', state_class=SensorStateClass.TOTAL_INCREASING, scope=FitbitScope.NUTRITION, entity_category=EntityCategory.DIAGNOSTIC), FitbitSensorEntityDescription(key='foods/log/water', translation_key='water', icon='mdi:cup-water', unit_fn=_water_unit, state_class=SensorStateClass.TOTAL_INCREASING, scope=FitbitScope.NUTRITION, entity_category=EntityCategory.DIAGNOSTIC))
SLEEP_START_TIME: FitbitSensorEntityDescription = FitbitSensorEntityDescription(key='sleep/startTime', translation_key='sleep_start_time', icon='mdi:clock', scope=FitbitScope.SLEEP, entity_category=EntityCategory.DIAGNOSTIC)
SLEEP_START_TIME_12HR: FitbitSensorEntityDescription = FitbitSensorEntityDescription(key='sleep/startTime', translation_key='sleep_start_time', icon='mdi:clock', value_fn=_clock_format_12h, scope=FitbitScope.SLEEP, entity_category=EntityCategory.DIAGNOSTIC)
FITBIT_RESOURCE_BATTERY: FitbitSensorEntityDescription = FitbitSensorEntityDescription(key='devices/battery', translation_key='battery', icon='mdi:battery', scope=FitbitScope.DEVICE, entity_category=EntityCategory.DIAG