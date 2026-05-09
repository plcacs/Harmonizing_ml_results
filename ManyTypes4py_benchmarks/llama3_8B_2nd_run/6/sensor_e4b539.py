from __future__ import annotations
from datetime import timedelta
from typing import Any, List

class QNAPSensor(CoordinatorEntity[QnapCoordinator], SensorEntity):
    """Base class for a QNAP sensor."""
    _attr_has_entity_name: bool = True

    def __init__(self, coordinator: QnapCoordinator, description: SensorEntityDescription, unique_id: str, monitor_device: str = None):
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entity_description: SensorEntityDescription = description
        self.device_name: str = self.coordinator.data['system_stats']['system']['name']
        self.monitor_device: str = monitor_device
        self._attr_unique_id: str = f'{unique_id}_{description.key}'
        if monitor_device:
            self._attr_unique_id = f'{self._attr_unique_id}_{monitor_device}'
            self._attr_translation_placeholders: dict = {'monitor_device': monitor_device}
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, unique_id)}, serial_number=unique_id, name=self.device_name, model=self.coordinator.data['system_stats']['system']['model'], sw_version=self.coordinator.data['system_stats']['firmware']['version'], manufacturer='QNAP')

class QNAPCPUSensor(QNAPSensor):
    """A QNAP sensor that monitors CPU stats."""

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        if self.entity_description.key == 'cpu_temp':
            return self.coordinator.data['system_stats']['cpu']['temp_c']
        if self.entity_description.key == 'cpu_usage':
            return self.coordinator.data['system_stats']['cpu']['usage_percent']
        return None

class QNAPMemorySensor(QNAPSensor):
    """A QNAP sensor that monitors memory stats."""

    @property
    def native_value(self) -> float:
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
    def native_value(self) -> str:
        """Return the state of the sensor."""
        nic: str = self.coordinator.data['system_stats']['nics'][self.monitor_device]
        if self.entity_description.key == 'network_link_status':
            return nic['link_status']
        if self.entity_description.key == 'network_max_speed':
            return nic['max_speed']
        if self.entity_description.key == 'network_err':
            return nic['err_packets']
        data: dict = self.coordinator.data['bandwidth'][self.monitor_device]
        if self.entity_description.key == 'network_tx':
            return data['tx']
        if self.entity_description.key == 'network_rx':
            return data['rx']
        return None

class QNAPSystemSensor(QNAPSensor):
    """A QNAP sensor that monitors overall system health."""

    @property
    def native_value(self) -> str:
        """Return the state of the sensor."""
        if self.entity_description.key == 'status':
            return self.coordinator.data['system_health']
        if self.entity_description.key == 'system_temp':
            return str(int(self.coordinator.data['system_stats']['system']['temp_c']))
        if self.entity_description.key == 'uptime':
            uptime: dict = self.coordinator.data['system_stats']['uptime']
            uptime_duration: timedelta = timedelta(days=uptime['days'], hours=uptime['hours'], minutes=uptime['minutes'], seconds=uptime['seconds'])
            return str(dt_util.now() - uptime_duration)
        return None

class QNAPDriveSensor(QNAPSensor):
    """A QNAP sensor that monitors HDD/SSD drive stats."""

    @property
    def native_value(self) -> str:
        """Return the state of the sensor."""
        data: dict = self.coordinator.data['smart_drive_health'][self.monitor_device]
        if self.entity_description.key == 'drive_smart_status':
            return data['health']
        if self.entity_description.key == 'drive_temp':
            return str(int(data['temp_c'])) if data['temp_c'] is not None else '0'
        return None

    @property
    def extra_state_attributes(self) -> dict:
        """Return the state attributes."""
        if self.coordinator.data:
            data: dict = self.coordinator.data['smart_drive_health'][self.monitor_device]
            return {ATTR_DRIVE: data['drive_number'], ATTR_MODEL: data['model'], ATTR_SERIAL: data['serial'], ATTR_TYPE: data['type']}
        return None

class QNAPVolumeSensor(QNAPSensor):
    """A QNAP sensor that monitors storage volume stats."""

    @property
    def native_value(self) -> str:
        """Return the state of the sensor."""
        data: dict = self.coordinator.data['volumes'][self.monitor_device]
        free_gb: int = int(data['free_size'])
        if self.entity_description.key == 'volume_size_free':
            return str(free_gb)
        total_gb: int = int(data['total_size'])
        if self.entity_description.key == 'volume_size_total':
            return str(total_gb)
        used_gb: int = total_gb - free_gb
        if self.entity_description.key == 'volume_size_used':
            return str(used_gb)
        if self.entity_description.key == 'volume_percentage_used':
            return str(int(used_gb / total_gb * 100))
        return None

SENSOR_KEYS: List[str] = [desc.key for desc in (*_SYSTEM_MON_COND, *_CPU_MON_COND, *_MEMORY_MON_COND, *_NETWORK_MON_COND, *_DRIVE_MON_COND, *_VOLUME_MON_COND)]
