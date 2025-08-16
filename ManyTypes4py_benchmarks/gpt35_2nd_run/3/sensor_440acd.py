from __future__ import annotations
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import ipaddress
import logging
import socket
import sys
import time
from typing import Any, Literal
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfDataRate, UnitOfInformation, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import slugify
from . import SystemMonitorConfigEntry
from .const import DOMAIN, NET_IO_TYPES
from .coordinator import SystemMonitorCoordinator
from .util import get_all_disk_mounts, get_all_network_interfaces, read_cpu_temperature

@lru_cache
def get_cpu_icon() -> str:
    ...

def get_network(entity: SystemMonitorSensor) -> float:
    ...

def get_packets(entity: SystemMonitorSensor) -> int:
    ...

def get_throughput(entity: SystemMonitorSensor) -> float:
    ...

def get_ip_address(entity: SystemMonitorSensor) -> str:
    ...

@dataclass(frozen=True, kw_only=True)
class SysMonitorSensorEntityDescription(SensorEntityDescription):
    ...

def check_legacy_resource(resource: str, resources: set) -> bool:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class SystemMonitorSensor(CoordinatorEntity[SystemMonitorCoordinator], SensorEntity):
    ...

    def __init__(self, coordinator: SystemMonitorCoordinator, sensor_description: SysMonitorSensorEntityDescription, entry_id: str, argument: str, legacy_enabled: bool = False) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    @property
    def available(self) -> bool:
        ...
