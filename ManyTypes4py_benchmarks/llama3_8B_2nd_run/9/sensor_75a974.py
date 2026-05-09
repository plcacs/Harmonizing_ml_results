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
_CONFIGURING: Final = {}
SCAN_INTERVAL: Final = datetime.timedelta(minutes=30)
FITBIT_TRACKER_SUBSTRING: Final = '/tracker/'

@dataclass(frozen=True)
class FitbitSensorEntityDescription(SensorEntityDescription):
    """Describes Fitbit sensor entity."""
    unit_type: None
    value_fn: Callable[[Any], str]
    unit_fn: Callable[[FitbitUnitSystem], UnitOfTime | UnitOfLength | UnitOfMass | UnitOfVolume]
    scope: FitbitScope

    @property
    def is_tracker(self) -> bool:
        """Return if the entity is a tracker."""
        return FITBIT_TRACKER_SUBSTRING in self.key

def _build_device_info(config_entry: FitbitConfigEntry, entity_description: FitbitSensorEntityDescription) -> DeviceInfo:
    """Build device info for sensor entities info across devices."""
    unique_id: str = cast(str, config_entry.unique_id)
    if entity_description.is_tracker:
        return DeviceInfo(entry_type=DeviceEntryType.SERVICE, identifiers={(DOMAIN, f'{unique_id}_tracker')}, translation_key='tracker', translation_placeholders={'display_name': config_entry.title})
    return DeviceInfo(entry_type=DeviceEntryType.SERVICE, identifiers={(DOMAIN, unique_id)})

FITBIT_RESOURCES_LIST: list[FitbitSensorEntityDescription] = [
    FitbitSensorEntityDescription(key='activities/activityCalories', translation_key='activity_calories', native_unit_of_measurement='cal', icon='mdi:fire', scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.TOTAL_INCREASING, entity_category=EntityCategory.DIAGNOSTIC),
    FitbitSensorEntityDescription(key='activities/calories', translation_key='calories', native_unit_of_measurement='cal', icon='mdi:fire', scope=FitbitScope.ACTIVITY, state_class=SensorStateClass.TOTAL_INCREASING),
    # ... and so on
]

SLEEP_START_TIME: FitbitSensorEntityDescription = FitbitSensorEntityDescription(key='sleep/startTime', translation_key='sleep_start_time', icon='mdi:clock', scope=FitbitScope.SLEEP, entity_category=EntityCategory.DIAGNOSTIC)
SLEEP_START_TIME_12HR: FitbitSensorEntityDescription = FitbitSensorEntityDescription(key='sleep/startTime', translation_key='sleep_start_time', icon='mdi:clock', value_fn=_clock_format_12h, scope=FitbitScope.SLEEP, entity_category=EntityCategory.DIAGNOSTIC)
FITBIT_RESOURCE_BATTERY: FitbitSensorEntityDescription = FitbitSensorEntityDescription(key='devices/battery', translation_key='battery', icon='mdi:battery', scope=FitbitScope.DEVICE, entity_category=EntityCategory.DIAGNOSTIC, has_entity_name=True)
FITBIT_RESOURCE_BATTERY_LEVEL: FitbitSensorEntityDescription = FitbitSensorEntityDescription(key='devices/battery_level', translation_key='battery_level', scope=FitbitScope.DEVICE, entity_category=EntityCategory.DIAGNOSTIC, has_entity_name=True, device_class=SensorDeviceClass.BATTERY, native_unit_of_measurement=PERCENTAGE)

async def async_setup_entry(hass: HomeAssistant, entry: FitbitConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
    """Set up the Fitbit sensor platform."""
    data: FitbitConfigEntry = entry.runtime_data
    api: FitbitApi = data.api
    user_profile: dict = await api.async_get_user_profile()
    unit_system: FitbitUnitSystem = await api.async_get_unit_system()
    fitbit_config: FitbitConfigEntry = config_from_entry_data(entry.data)

    def is_explicit_enable(description: FitbitSensorEntityDescription) -> bool:
        """Determine if entity is enabled by default."""
        return fitbit_config.is_explicit_enable(description.key)

    def is_allowed_resource(description: FitbitSensorEntityDescription) -> bool:
        """Determine if an entity is allowed to be created."""
        return fitbit_config.is_allowed_resource(description.scope, description.key)

    resource_list: list[FitbitSensorEntityDescription] = [*FITBIT_RESOURCES_LIST, SLEEP_START_TIME_12HR if fitbit_config.clock_format == '12H' else SLEEP_START_TIME]
    entities: list[FitbitSensor] = [
        FitbitSensor(entry, api, user_profile['encoded_id'], description, units=description.unit_fn(unit_system), enable_default_override=is_explicit_enable(description), device_info=_build_device_info(entry, description)) 
        for description in resource_list if is_allowed_resource(description)
    ]
    async_add_entities(entities)
    if data.device_coordinator and is_allowed_resource(FITBIT_RESOURCE_BATTERY):
        battery_entities: list[FitbitBatterySensor] = [
            FitbitBatterySensor(data.device_coordinator, user_profile['encoded_id'], FITBIT_RESOURCE_BATTERY, device=device, enable_default_override=is_explicit_enable(FITBIT_RESOURCE_BATTERY)) 
            for device in data.device_coordinator.data.values()
        ]
        battery_entities.extend((FitbitBatteryLevelSensor(data.device_coordinator, user_profile['encoded_id'], FITBIT_RESOURCE_BATTERY_LEVEL, device=device) for device in data.device_coordinator.data.values()))
        async_add_entities(battery_entities)

class FitbitSensor(SensorEntity):
    """Implementation of a Fitbit sensor."""
    _attr_attribution: str = ATTRIBUTION
    _attr_has_entity_name: bool = True

    def __init__(self, config_entry: FitbitConfigEntry, api: FitbitApi, user_profile_id: str, description: FitbitSensorEntityDescription, units: UnitOfTime | UnitOfLength | UnitOfMass | UnitOfVolume, enable_default_override: bool, device_info: DeviceInfo):
        """Initialize the Fitbit sensor."""
        self.config_entry: FitbitConfigEntry = config_entry
        self.entity_description: FitbitSensorEntityDescription = description
        self.api: FitbitApi = api
        self._attr_unique_id: str = f'{user_profile_id}_{description.key}'
        self._attr_device_info: DeviceInfo = device_info
        if units is not None:
            self._attr_native_unit_of_measurement: UnitOfTime | UnitOfLength | UnitOfMass | UnitOfVolume = units
        if enable_default_override:
            self._attr_entity_registry_enabled_default: bool = True

    async def async_update(self) -> None:
        """Get the latest data from the Fitbit API and update the states."""
        try:
            result: dict = await self.api.async_get_latest_time_series(self.entity_description.key)
        except FitbitAuthException:
            self._attr_available: bool = False
            self.config_entry.async_start_reauth(self.hass)
        except FitbitApiException:
            self._attr_available: bool = False
        else:
            self._attr_available: bool = True
            self._attr_native_value: str = self.entity_description.value_fn(result)

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self.async_schedule_update_ha_state(force_refresh=True)

class FitbitBatterySensor(CoordinatorEntity[FitbitDeviceCoordinator], SensorEntity):
    """Implementation of a Fitbit battery sensor."""
    _attr_attribution: str = ATTRIBUTION

    def __init__(self, coordinator: FitbitDeviceCoordinator, user_profile_id: str, description: FitbitSensorEntityDescription, device: FitbitDevice, enable_default_override: bool):
        """Initialize the Fitbit sensor."""
        super().__init__(coordinator)
        self.entity_description: FitbitSensorEntityDescription = description
        self.device: FitbitDevice = device
        self._attr_unique_id: str = f'{user_profile_id}_{description.key}_{device.id}'
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, f'{user_profile_id}_{device.id}')}, name=device.device_version, model=device.device_version)
        if enable_default_override:
            self._attr_entity_registry_enabled_default: bool = True

    @property
    def icon(self) -> str:
        """Icon to use in the frontend, if any."""
        if (battery_level := BATTERY_LEVELS.get(self.device.battery)):
            return icon_for_battery_level(battery_level=battery_level)
        return self.entity_description.icon

    @property
    def extra_state_attributes(self) -> dict:
        """Return the state attributes."""
        return {'model': self.device.device_version, 'type': self.device.type.lower() if self.device.type is not None else None}

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass update state from existing coordinator data."""
        await super().async_added_to_hass()
        self._handle_coordinator_update()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.device = self.coordinator.data[self.device.id]
        self._attr_native_value: float = self.device.battery
        self.async_write_ha_state()

class FitbitBatteryLevelSensor(CoordinatorEntity[FitbitDeviceCoordinator], SensorEntity):
    """Implementation of a Fitbit battery level sensor."""
    _attr_attribution: str = ATTRIBUTION

    def __init__(self, coordinator: FitbitDeviceCoordinator, user_profile_id: str, description: FitbitSensorEntityDescription, device: FitbitDevice):
        """Initialize the Fitbit sensor."""
        super().__init__(coordinator)
        self.entity_description: FitbitSensorEntityDescription = description
        self.device: FitbitDevice = device
        self._attr_unique_id: str = f'{user_profile_id}_{description.key}_{device.id}'
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, f'{user_profile_id}_{device.id}')}, name=device.device_version, model=device.device_version)

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass update state from existing coordinator data."""
        await super().async_added_to_hass()
        self._handle_coordinator_update()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.device = self.coordinator.data[self.device.id]
        self._attr_native_value: float = self.device.battery_level
        self.async_write_ha_state()