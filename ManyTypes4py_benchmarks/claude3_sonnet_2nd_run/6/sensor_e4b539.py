"""Support for QNAP NAS Sensors."""
from __future__ import annotations
from datetime import timedelta
from typing import Any, Final
from homeassistant import config_entries
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfDataRate, UnitOfInformation, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
from .const import DOMAIN
from .coordinator import QnapCoordinator

ATTR_DRIVE: Final = 'Drive'
ATTR_IP: Final = 'IP Address'
ATTR_MAC: Final = 'MAC Address'
ATTR_MASK: Final = 'Mask'
ATTR_MAX_SPEED: Final = 'Max Speed'
ATTR_MEMORY_SIZE: Final = 'Memory Size'
ATTR_MODEL: Final = 'Model'
ATTR_PACKETS_ERR: Final = 'Packets (Err)'
ATTR_SERIAL: Final = 'Serial #'
ATTR_TYPE: Final = 'Type'
ATTR_UPTIME: Final = 'Uptime'
ATTR_VOLUME_SIZE: Final = 'Volume Size'

_SYSTEM_MON_COND: Final[tuple[SensorEntityDescription, ...]] = (
    SensorEntityDescription(key='status', translation_key='status', entity_category=EntityCategory.DIAGNOSTIC),
    SensorEntityDescription(key='system_temp', translation_key='system_temp', native_unit_of_measurement=UnitOfTemperature.CELSIUS, device_class=SensorDeviceClass.TEMPERATURE, entity_category=EntityCategory.DIAGNOSTIC, state_class=SensorStateClass.MEASUREMENT),
    SensorEntityDescription(key='uptime', translation_key='uptime', device_class=SensorDeviceClass.TIMESTAMP, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False)
)

_CPU_MON_COND: Final[tuple[SensorEntityDescription, ...]] = (
    SensorEntityDescription(key='cpu_temp', translation_key='cpu_temp', native_unit_of_measurement=UnitOfTemperature.CELSIUS, device_class=SensorDeviceClass.TEMPERATURE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT),
    SensorEntityDescription(key='cpu_usage', translation_key='cpu_usage', native_unit_of_measurement=PERCENTAGE, entity_category=EntityCategory.DIAGNOSTIC, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=0)
)

_MEMORY_MON_COND: Final[tuple[SensorEntityDescription, ...]] = (
    SensorEntityDescription(key='memory_size', translation_key='memory_size', native_unit_of_measurement=UnitOfInformation.MEBIBYTES, device_class=SensorDeviceClass.DATA_SIZE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES),
    SensorEntityDescription(key='memory_free', translation_key='memory_free', native_unit_of_measurement=UnitOfInformation.MEBIBYTES, device_class=SensorDeviceClass.DATA_SIZE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES),
    SensorEntityDescription(key='memory_used', translation_key='memory_used', native_unit_of_measurement=UnitOfInformation.MEBIBYTES, device_class=SensorDeviceClass.DATA_SIZE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES),
    SensorEntityDescription(key='memory_percent_used', translation_key='memory_percent_used', native_unit_of_measurement=PERCENTAGE, entity_category=EntityCategory.DIAGNOSTIC, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=0)
)

_NETWORK_MON_COND: Final[tuple[SensorEntityDescription, ...]] = (
    SensorEntityDescription(key='network_link_status', translation_key='network_link_status', entity_category=EntityCategory.DIAGNOSTIC),
    SensorEntityDescription(key='network_tx', translation_key='network_tx', native_unit_of_measurement=UnitOfDataRate.BITS_PER_SECOND, device_class=SensorDeviceClass.DATA_RATE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfDataRate.MEGABITS_PER_SECOND),
    SensorEntityDescription(key='network_rx', translation_key='network_rx', native_unit_of_measurement=UnitOfDataRate.BITS_PER_SECOND, device_class=SensorDeviceClass.DATA_RATE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfDataRate.MEGABITS_PER_SECOND),
    SensorEntityDescription(key='network_err', translation_key='network_err', entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT),
    SensorEntityDescription(key='network_max_speed', translation_key='network_max_speed', native_unit_of_measurement=UnitOfDataRate.MEGABITS_PER_SECOND, device_class=SensorDeviceClass.DATA_RATE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT)
)

_DRIVE_MON_COND: Final[tuple[SensorEntityDescription, ...]] = (
    SensorEntityDescription(key='drive_smart_status', translation_key='drive_smart_status', entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False),
    SensorEntityDescription(key='drive_temp', translation_key='drive_temp', native_unit_of_measurement=UnitOfTemperature.CELSIUS, device_class=SensorDeviceClass.TEMPERATURE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT)
)

_VOLUME_MON_COND: Final[tuple[SensorEntityDescription, ...]] = (
    SensorEntityDescription(key='volume_size_total', translation_key='volume_size_total', native_unit_of_measurement=UnitOfInformation.BYTES, device_class=SensorDeviceClass.DATA_SIZE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES),
    SensorEntityDescription(key='volume_size_used', translation_key='volume_size_used', native_unit_of_measurement=UnitOfInformation.BYTES, device_class=SensorDeviceClass.DATA_SIZE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES),
    SensorEntityDescription(key='volume_size_free', translation_key='volume_size_free', native_unit_of_measurement=UnitOfInformation.BYTES, device_class=SensorDeviceClass.DATA_SIZE, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=1, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES),
    SensorEntityDescription(key='volume_percentage_used', translation_key='volume_percentage_used', native_unit_of_measurement=PERCENTAGE, entity_category=EntityCategory.DIAGNOSTIC, state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=0)
)

SENSOR_KEYS: Final[list[str]] = [desc.key for desc in (*_SYSTEM_MON_COND, *_CPU_MON_COND, *_MEMORY_MON_COND, *_NETWORK_MON_COND, *_DRIVE_MON_COND, *_VOLUME_MON_COND)]

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: config_entries.ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up entry."""
    coordinator: QnapCoordinator = QnapCoordinator(hass, config_entry)
    await coordinator.async_refresh()
    if not coordinator.last_update_success:
        raise PlatformNotReady
    uid = config_entry.unique_id
    assert uid is not None
    sensors: list[SensorEntity] = []
    sensors.extend([QNAPSystemSensor(coordinator, description, uid) for description in _SYSTEM_MON_COND])
    sensors.extend([QNAPCPUSensor(coordinator, description, uid) for description in _CPU_MON_COND])
    sensors.extend([QNAPMemorySensor(coordinator, description, uid) for description in _MEMORY_MON_COND])
    sensors.extend([QNAPNetworkSensor(coordinator, description, uid, nic) for nic in coordinator.data['system_stats']['nics'] for description in _NETWORK_MON_COND])
    sensors.extend([QNAPDriveSensor(coordinator, description, uid, drive) for drive in coordinator.data['smart_drive_health'] for description in _DRIVE_MON_COND])
    sensors.extend([QNAPVolumeSensor(coordinator, description, uid, volume) for volume in coordinator.data['volumes'] for description in _VOLUME_MON_COND])
    async_add_entities(sensors)

class QNAPSensor(CoordinatorEntity[QnapCoordinator], SensorEntity):
    """Base class for a QNAP sensor."""
    _attr_has_entity_name: bool = True

    def __init__(
        self,
        coordinator: QnapCoordinator,
        description: SensorEntityDescription,
        unique_id: str,
        monitor_device: str | None = None,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entity_description = description
        self.device_name: str = self.coordinator.data['system_stats']['system']['name']
        self.monitor_device: str | None = monitor_device
        self._attr_unique_id: str = f'{unique_id}_{description.key}'
        if monitor_device:
            self._attr_unique_id = f'{self._attr_unique_id}_{monitor_device}'
            self._attr_translation_placeholders = {'monitor_device': monitor_device}
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, unique_id)},
            serial_number=unique_id,
            name=self.device_name,
            model=self.coordinator.data['system_stats']['system']['model'],
            sw_version=self.coordinator.data['system_stats']['firmware']['version'],
            manufacturer='QNAP'
        )

class QNAPCPUSensor(QNAPSensor):
    """A QNAP sensor that monitors CPU stats."""

    @property
    def native_value(self) -> float | None:
        """Return the state of the sensor."""
        if self.entity_description.key == 'cpu_temp':
            return self.coordinator.data['system_stats']['cpu']['temp_c']
        if self.entity_description.key == 'cpu_usage':
            return self.coordinator.data['system_stats']['cpu']['usage_percent']
        return None

class QNAPMemorySensor(QNAPSensor):
    """A QNAP sensor that monitors memory stats."""

    @property
    def native_value(self) -> float | None:
        """Return the state of the sensor."""
        free: float = float(self.coordinator.data['system_stats']['memory']['free'])
        if self.entity_description.key == 'memory_free':
            return free
        total: float = float(self.coordinator.data['system_stats']['memory']['total'])
        if self.entity_description.key == 'memory_size':
            return total
        used: float = total - free
        if self.entity_description.key == 'memory_used':
            return used
        if self.entity_description.key == 'memory_percent_used':
            return used / total * 100
        return None

class QNAPNetworkSensor(QNAPSensor):
    """A QNAP sensor that monitors network stats."""

    @property
    def native_value(self) -> str | int | float | None:
        """Return the state of the sensor."""
        nic: dict[str, Any] = self.coordinator.data['system_stats']['nics'][self.monitor_device]
        if self.entity_description.key == 'network_link_status':
            return nic['link_status']
        if self.entity_description.key == 'network_max_speed':
            return nic['max_speed']
        if self.entity_description.key == 'network_err':
            return nic['err_packets']
        data: dict[str, Any] = self.coordinator.data['bandwidth'][self.monitor_device]
        if self.entity_description.key == 'network_tx':
            return data['tx']
        if self.entity_description.key == 'network_rx':
            return data['rx']
        return None

class QNAPSystemSensor(QNAPSensor):
    """A QNAP sensor that monitors overall system health."""

    @property
    def native_value(self) -> str | int | float | None:
        """Return the state of the sensor."""
        if self.entity_description.key == 'status':
            return self.coordinator.data['system_health']
        if self.entity_description.key == 'system_temp':
            return int(self.coordinator.data['system_stats']['system']['temp_c'])
        if self.entity_description.key == 'uptime':
            uptime: dict[str, int] = self.coordinator.data['system_stats']['uptime']
            uptime_duration: timedelta = timedelta(days=uptime['days'], hours=uptime['hours'], minutes=uptime['minutes'], seconds=uptime['seconds'])
            return dt_util.now() - uptime_duration
        return None

class QNAPDriveSensor(QNAPSensor):
    """A QNAP sensor that monitors HDD/SSD drive stats."""

    @property
    def native_value(self) -> str | int | None:
        """Return the state of the sensor."""
        data: dict[str, Any] = self.coordinator.data['smart_drive_health'][self.monitor_device]
        if self.entity_description.key == 'drive_smart_status':
            return data['health']
        if self.entity_description.key == 'drive_temp':
            return int(data['temp_c']) if data['temp_c'] is not None else 0
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the state attributes."""
        if self.coordinator.data:
            data: dict[str, Any] = self.coordinator.data['smart_drive_health'][self.monitor_device]
            return {ATTR_DRIVE: data['drive_number'], ATTR_MODEL: data['model'], ATTR_SERIAL: data['serial'], ATTR_TYPE: data['type']}
        return None

class QNAPVolumeSensor(QNAPSensor):
    """A QNAP sensor that monitors storage volume stats."""

    @property
    def native_value(self) -> int | float | None:
        """Return the state of the sensor."""
        data: dict[str, Any] = self.coordinator.data['volumes'][self.monitor_device]
        free_gb: int = int(data['free_size'])
        if self.entity_description.key == 'volume_size_free':
            return free_gb
        total_gb: int = int(data['total_size'])
        if self.entity_description.key == 'volume_size_total':
            return total_gb
        used_gb: int = total_gb - free_gb
        if self.entity_description.key == 'volume_size_used':
            return used_gb
        if self.entity_description.key == 'volume_percentage_used':
            return used_gb / total_gb * 100
        return None
