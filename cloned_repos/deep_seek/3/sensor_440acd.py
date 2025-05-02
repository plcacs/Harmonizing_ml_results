"""Support for monitoring the local system."""
from __future__ import annotations
from collections.abc import Callable, Set
import contextlib
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import ipaddress
import logging
import socket
import sys
import time
from typing import Any, Literal, Optional, Dict, List, Tuple, Union, cast

from homeassistant.components.sensor import (
    DOMAIN as SENSOR_DOMAIN, 
    SensorDeviceClass, 
    SensorEntity, 
    SensorEntityDescription, 
    SensorStateClass
)
from homeassistant.const import (
    PERCENTAGE, 
    EntityCategory, 
    UnitOfDataRate, 
    UnitOfInformation, 
    UnitOfTemperature
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import slugify

from . import SystemMonitorConfigEntry
from .const import DOMAIN, NET_IO_TYPES
from .coordinator import SystemMonitorCoordinator
from .util import get_all_disk_mounts, get_all_network_interfaces, read_cpu_temperature

_LOGGER: logging.Logger = logging.getLogger(__name__)
CONF_ARG: str = 'arg'
SENSOR_TYPE_NAME: int = 0
SENSOR_TYPE_UOM: int = 1
SENSOR_TYPE_ICON: int = 2
SENSOR_TYPE_DEVICE_CLASS: int = 3
SENSOR_TYPE_MANDATORY_ARG: int = 4
SIGNAL_SYSTEMMONITOR_UPDATE: str = 'systemmonitor_update'

@lru_cache
def get_cpu_icon() -> str:
    """Return cpu icon."""
    if sys.maxsize > 2 ** 32:
        return 'mdi:cpu-64-bit'
    return 'mdi:cpu-32-bit'

def get_network(entity: 'SystemMonitorSensor') -> Optional[float]:
    """Return network in and out."""
    counters = entity.coordinator.data.io_counters
    if entity.argument in counters:
        counter = counters[entity.argument][IO_COUNTER[entity.entity_description.key]]
        return round(counter / 1024 ** 2, 1)
    return None

def get_packets(entity: 'SystemMonitorSensor') -> Optional[int]:
    """Return packets in and out."""
    counters = entity.coordinator.data.io_counters
    if entity.argument in counters:
        return counters[entity.argument][IO_COUNTER[entity.entity_description.key]]
    return None

def get_throughput(entity: 'SystemMonitorSensor') -> Optional[float]:
    """Return network throughput in and out."""
    counters = entity.coordinator.data.io_counters
    state = None
    if entity.argument in counters:
        counter = counters[entity.argument][IO_COUNTER[entity.entity_description.key]]
        now = time.monotonic()
        if (value := entity.value) and (update_time := entity.update_time) and (value < counter):
            state = round((counter - value) / 1000 ** 2 / (now - update_time), 3)
        entity.update_time = now
        entity.value = counter
    return state

def get_ip_address(entity: 'SystemMonitorSensor') -> Optional[str]:
    """Return network ip address."""
    addresses = entity.coordinator.data.addresses
    if entity.argument in addresses:
        for addr in addresses[entity.argument]:
            if addr.family == IF_ADDRS_FAMILY[entity.entity_description.key]:
                address = ipaddress.ip_address(addr.address)
                if address.version == 6 and (address.is_link_local or address.is_loopback):
                    continue
                return addr.address
    return None

@dataclass(frozen=True, kw_only=True)
class SysMonitorSensorEntityDescription(SensorEntityDescription):
    """Describes System Monitor sensor entities."""
    none_is_unavailable: bool = False
    mandatory_arg: bool = False
    placeholder: Optional[str] = None
    value_fn: Callable[['SystemMonitorSensor'], Optional[Union[float, int, str, datetime]]] = lambda _: None
    add_to_update: Callable[['SystemMonitorSensor'], Tuple[str, str]] = lambda _: ('', '')

SENSOR_TYPES: Dict[str, SysMonitorSensorEntityDescription] = {
    'disk_free': SysMonitorSensorEntityDescription(
        key='disk_free', 
        translation_key='disk_free', 
        placeholder='mount_point', 
        native_unit_of_measurement=UnitOfInformation.GIBIBYTES, 
        device_class=SensorDeviceClass.DATA_SIZE, 
        state_class=SensorStateClass.MEASUREMENT, 
        value_fn=lambda entity: round(entity.coordinator.data.disk_usage[entity.argument].free / 1024 ** 3, 1) if entity.argument in entity.coordinator.data.disk_usage else None, 
        none_is_unavailable=True, 
        add_to_update=lambda entity: ('disks', entity.argument)
    ),
    # ... (rest of the SENSOR_TYPES dictionary remains the same)
}

def check_legacy_resource(resource: str, resources: Set[str]) -> bool:
    """Return True if legacy resource was configured."""
    if resource in resources:
        _LOGGER.debug('Checking %s in %s returns True', resource, ', '.join(resources))
        return True
    _LOGGER.debug('Checking %s in %s returns False', resource, ', '.join(resources))
    return False

IO_COUNTER: Dict[str, int] = {
    'network_out': 0, 
    'network_in': 1, 
    'packets_out': 2, 
    'packets_in': 3, 
    'throughput_network_out': 0, 
    'throughput_network_in': 1
}

IF_ADDRS_FAMILY: Dict[str, int] = {
    'ipv4_address': socket.AF_INET, 
    'ipv6_address': socket.AF_INET6
}

async def async_setup_entry(
    hass: HomeAssistant,
    entry: SystemMonitorConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up System Monitor sensors based on a config entry."""
    entities: List[SystemMonitorSensor] = []
    legacy_resources: Set[str] = set(entry.options.get('resources', []))
    loaded_resources: Set[str] = set()
    coordinator: SystemMonitorCoordinator = entry.runtime_data.coordinator
    psutil_wrapper = entry.runtime_data.psutil_wrapper
    sensor_data = coordinator.data

    def get_arguments() -> Dict[str, Any]:
        """Return startup information."""
        return {
            'disk_arguments': get_all_disk_mounts(hass, psutil_wrapper),
            'network_arguments': get_all_network_interfaces(hass, psutil_wrapper)
        }

    cpu_temperature: Optional[float] = None
    with contextlib.suppress(AttributeError):
        cpu_temperature = read_cpu_temperature(sensor_data.temperatures)
    
    startup_arguments: Dict[str, Any] = await hass.async_add_executor_job(get_arguments)
    startup_arguments['cpu_temperature'] = cpu_temperature
    
    _LOGGER.debug('Setup from options %s', entry.options)
    
    # ... (rest of the async_setup_entry function remains the same)

class SystemMonitorSensor(CoordinatorEntity[SystemMonitorCoordinator], SensorEntity):
    """Implementation of a system monitor sensor."""
    _attr_has_entity_name: bool = True
    _attr_entity_category: EntityCategory = EntityCategory.DIAGNOSTIC
    value: Optional[Union[float, int]] = None
    update_time: Optional[float] = None

    def __init__(
        self,
        coordinator: SystemMonitorCoordinator,
        sensor_description: SysMonitorSensorEntityDescription,
        entry_id: str,
        argument: str,
        legacy_enabled: bool = False
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entity_description: SysMonitorSensorEntityDescription = sensor_description
        if self.entity_description.placeholder:
            self._attr_translation_placeholders: Dict[str, str] = {
                self.entity_description.placeholder: argument
            }
        self._attr_unique_id: str = slugify(f'{sensor_description.key}_{argument}')
        self._attr_entity_registry_enabled_default: bool = legacy_enabled
        self._attr_device_info: DeviceInfo = DeviceInfo(
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, entry_id)},
            manufacturer='System Monitor',
            name='System Monitor'
        )
        self.argument: str = argument
        self._attr_native_value: Optional[Union[float, int, str, datetime]] = self.entity_description.value_fn(self)

    async def async_added_to_hass(self) -> None:
        """When added to hass."""
        self.coordinator.update_subscribers[self.entity_description.add_to_update(self)].add(self.entity_id)
        return await super().async_added_to_hass()

    async def async_will_remove_from_hass(self) -> None:
        """When removed from hass."""
        self.coordinator.update_subscribers[self.entity_description.add_to_update(self)].remove(self.entity_id)
        return await super().async_will_remove_from_hass()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self._attr_native_value = self.entity_description.value_fn(self)
        super()._handle_coordinator_update()

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        if self.entity_description.none_is_unavailable:
            return self.coordinator.last_update_success is True and self.native_value is not None
        return super().available
