from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast, List, Optional
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_DISKS, PERCENTAGE, EntityCategory, UnitOfDataRate, UnitOfInformation, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from .coordinator import SynologyDSMCentralUpdateCoordinator
from .entity import SynologyDSMBaseEntity, SynologyDSMDeviceEntity, SynologyDSMEntityDescription
from .models import SynologyDSMData

@dataclass(frozen=True, kw_only=True)
class SynologyDSMSensorEntityDescription(SensorEntityDescription, SynologyDSMEntityDescription):
    """Describes Synology DSM sensor entity."""

UTILISATION_SENSORS: List[SynologyDSMSensorEntityDescription]
STORAGE_VOL_SENSORS: List[SynologyDSMSensorEntityDescription]
STORAGE_DISK_SENSORS: List[SynologyDSMSensorEntityDescription]
INFORMATION_SENSORS: List[SynologyDSMSensorEntityDescription]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class SynoDSMSensor(SynologyDSMBaseEntity[SynologyDSMCentralUpdateCoordinator], SensorEntity):
    """Mixin for sensor specific attributes."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMSensorEntityDescription) -> None:

class SynoDSMUtilSensor(SynoDSMSensor):
    """Representation a Synology Utilisation sensor."""

    @property
    def native_value(self) -> StateType:

    @property
    def available(self) -> bool:

class SynoDSMStorageSensor(SynologyDSMDeviceEntity, SynoDSMSensor):
    """Representation a Synology Storage sensor."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMSensorEntityDescription, device_id: Optional[str] = None) -> None:

    @property
    def native_value(self) -> StateType:

class SynoDSMInfoSensor(SynoDSMSensor):
    """Representation a Synology information sensor."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMSensorEntityDescription) -> None:

    @property
    def native_value(self) -> StateType:
