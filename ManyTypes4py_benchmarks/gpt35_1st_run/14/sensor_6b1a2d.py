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
    api_key: str
    key: str
    translation_key: str
    native_unit_of_measurement: str
    suggested_unit_of_measurement: Optional[str] = None
    suggested_display_precision: Optional[int] = None
    device_class: Optional[str] = None
    entity_category: Optional[str] = None
    state_class: Optional[str] = None
    entity_registry_enabled_default: Optional[bool] = None

UTILISATION_SENSORS: List[SynologyDSMSensorEntityDescription]
STORAGE_VOL_SENSORS: List[SynologyDSMSensorEntityDescription]
STORAGE_DISK_SENSORS: List[SynologyDSMSensorEntityDescription]
INFORMATION_SENSORS: List[SynologyDSMSensorEntityDescription]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Synology NAS Sensor."""
    data = hass.data[DOMAIN][entry.unique_id]
    api = data.api
    coordinator = data.coordinator_central
    storage = api.storage
    assert storage is not None
    entities: List[SynoDSMSensor] = []
    if storage.volumes_ids:
        entities.extend([SynoDSMStorageSensor(api, coordinator, description, volume) for volume in entry.data.get(CONF_VOLUMES, storage.volumes_ids) for description in STORAGE_VOL_SENSORS])
    if storage.disks_ids:
        entities.extend([SynoDSMStorageSensor(api, coordinator, description, disk) for disk in entry.data.get(CONF_DISKS, storage.disks_ids) for description in STORAGE_DISK_SENSORS])
    entities.extend([SynoDSMInfoSensor(api, coordinator, description) for description in INFORMATION_SENSORS])
    async_add_entities(entities)

class SynoDSMSensor(SynologyDSMBaseEntity[SynologyDSMCentralUpdateCoordinator], SensorEntity):
    """Mixin for sensor specific attributes."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMSensorEntityDescription) -> None:
        """Initialize the Synology DSM sensor entity."""
        super().__init__(api, coordinator, description)

class SynoDSMUtilSensor(SynoDSMSensor):
    """Representation a Synology Utilisation sensor."""

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        attr = getattr(self._api.utilisation, self.entity_description.key)
        if callable(attr):
            attr = attr()
        if isinstance(attr, int) and self.native_unit_of_measurement == ENTITY_UNIT_LOAD:
            return round(attr / 100, 2)
        return attr

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return bool(self._api.utilisation) and super().available

class SynoDSMStorageSensor(SynologyDSMDeviceEntity, SynoDSMSensor):
    """Representation a Synology Storage sensor."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMSensorEntityDescription, device_id: Optional[str] = None) -> None:
        """Initialize the Synology DSM storage sensor entity."""
        super().__init__(api, coordinator, description, device_id)

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        return cast(StateType, getattr(self._api.storage, self.entity_description.key)(self._device_id))

class SynoDSMInfoSensor(SynoDSMSensor):
    """Representation a Synology information sensor."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMSensorEntityDescription) -> None:
        """Initialize the Synology SynoDSMInfoSensor entity."""
        super().__init__(api, coordinator, description)
        self._previous_uptime = None
        self._last_boot = None

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        attr = getattr(self._api.information, self.entity_description.key)
        if attr is None:
            return None
        if self.entity_description.key == 'uptime':
            if self._previous_uptime is None or self._previous_uptime > attr:
                self._last_boot = utcnow() - timedelta(seconds=attr)
            self._previous_uptime = attr
            return self._last_boot
        return attr
