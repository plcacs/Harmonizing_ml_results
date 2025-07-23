"""Support for Synology DSM sensors."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast, Any, Callable

from synology_dsm.api.core.utilization import SynoCoreUtilization
from synology_dsm.api.dsm.information import SynoDSMInformation
from synology_dsm.api.storage.storage import SynoStorage
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_DISKS, PERCENTAGE, EntityCategory, UnitOfDataRate, UnitOfInformation, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util.dt import utcnow

from . import SynoApi
from .const import CONF_VOLUMES, DOMAIN, ENTITY_UNIT_LOAD
from .coordinator import SynologyDSMCentralUpdateCoordinator
from .entity import SynologyDSMBaseEntity, SynologyDSMDeviceEntity, SynologyDSMEntityDescription
from .models import SynologyDSMData

@dataclass(frozen=True, kw_only=True)
class SynologyDSMSensorEntityDescription(SensorEntityDescription, SynologyDSMEntityDescription):
    """Describes Synology DSM sensor entity."""

UTILISATION_SENSORS: tuple[SynologyDSMSensorEntityDescription, ...] = (
    SynologyDSMSensorEntityDescription(
        api_key=SynoCoreUtilization.API_KEY,
        key='cpu_other_load',
        translation_key='cpu_other_load',
        native_unit_of_measurement=PERCENTAGE,
        entity_registry_enabled_default=False,
        state_class=SensorStateClass.MEASUREMENT
    ),
    # ... (rest of the UTILISATION_SENSORS definitions remain the same)
)

STORAGE_VOL_SENSORS: tuple[SynologyDSMSensorEntityDescription, ...] = (
    SynologyDSMSensorEntityDescription(
        api_key=SynoStorage.API_KEY,
        key='volume_status',
        translation_key='volume_status'
    ),
    # ... (rest of the STORAGE_VOL_SENSORS definitions remain the same)
)

STORAGE_DISK_SENSORS: tuple[SynologyDSMSensorEntityDescription, ...] = (
    SynologyDSMSensorEntityDescription(
        api_key=SynoStorage.API_KEY,
        key='disk_smart_status',
        translation_key='disk_smart_status',
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.DIAGNOSTIC
    ),
    # ... (rest of the STORAGE_DISK_SENSORS definitions remain the same)
)

INFORMATION_SENSORS: tuple[SynologyDSMSensorEntityDescription, ...] = (
    SynologyDSMSensorEntityDescription(
        api_key=SynoDSMInformation.API_KEY,
        key='temperature',
        translation_key='temperature',
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC
    ),
    # ... (rest of the INFORMATION_SENSORS definitions remain the same)
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Synology NAS Sensor."""
    data: SynologyDSMData = hass.data[DOMAIN][entry.unique_id]
    api: SynoApi = data.api
    coordinator: SynologyDSMCentralUpdateCoordinator = data.coordinator_central
    storage: SynoStorage | None = api.storage
    assert storage is not None

    entities: list[SensorEntity] = [
        SynoDSMUtilSensor(api, coordinator, description)
        for description in UTILISATION_SENSORS
    ]

    if storage.volumes_ids:
        entities.extend([
            SynoDSMStorageSensor(api, coordinator, description, volume)
            for volume in entry.data.get(CONF_VOLUMES, storage.volumes_ids)
            for description in STORAGE_VOL_SENSORS
        ])

    if storage.disks_ids:
        entities.extend([
            SynoDSMStorageSensor(api, coordinator, description, disk)
            for disk in entry.data.get(CONF_DISKS, storage.disks_ids)
            for description in STORAGE_DISK_SENSORS
        ])

    entities.extend([
        SynoDSMInfoSensor(api, coordinator, description)
        for description in INFORMATION_SENSORS
    ])

    async_add_entities(entities)

class SynoDSMSensor(SynologyDSMBaseEntity[SynologyDSMCentralUpdateCoordinator], SensorEntity):
    """Mixin for sensor specific attributes."""

    def __init__(
        self,
        api: SynoApi,
        coordinator: SynologyDSMCentralUpdateCoordinator,
        description: SynologyDSMSensorEntityDescription,
    ) -> None:
        """Initialize the Synology DSM sensor entity."""
        super().__init__(api, coordinator, description)

class SynoDSMUtilSensor(SynoDSMSensor):
    """Representation a Synology Utilisation sensor."""

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        attr: Any = getattr(self._api.utilisation, self.entity_description.key)
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

    def __init__(
        self,
        api: SynoApi,
        coordinator: SynologyDSMCentralUpdateCoordinator,
        description: SynologyDSMSensorEntityDescription,
        device_id: str | None = None,
    ) -> None:
        """Initialize the Synology DSM storage sensor entity."""
        super().__init__(api, coordinator, description, device_id)

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        return cast(StateType, getattr(self._api.storage, self.entity_description.key)(self._device_id))

class SynoDSMInfoSensor(SynoDSMSensor):
    """Representation a Synology information sensor."""

    def __init__(
        self,
        api: SynoApi,
        coordinator: SynologyDSMCentralUpdateCoordinator,
        description: SynologyDSMSensorEntityDescription,
    ) -> None:
        """Initialize the Synology SynoDSMInfoSensor entity."""
        super().__init__(api, coordinator, description)
        self._previous_uptime: int | None = None
        self._last_boot: datetime | None = None

    @property
    def native_value(self) -> StateType:
        """Return the state."""
        attr: Any = getattr(self._api.information, self.entity_description.key)
        if attr is None:
            return None
        if self.entity_description.key == 'uptime':
            if self._previous_uptime is None or self._previous_uptime > attr:
                self._last_boot = utcnow() - timedelta(seconds=attr)
            self._previous_uptime = attr
            return self._last_boot
        return attr
