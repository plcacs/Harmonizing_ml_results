from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any, Dict, Callable, Optional
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity, BinarySensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_DISKS, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .coordinator import SynologyDSMCentralUpdateCoordinator
from .entity import SynologyDSMBaseEntity, SynologyDSMDeviceEntity, SynologyDSMEntityDescription
from .models import SynologyDSMData

@dataclass(frozen=True, kw_only=True)
class SynologyDSMBinarySensorEntityDescription(BinarySensorEntityDescription, SynologyDSMEntityDescription):
    """Describes Synology DSM binary sensor entity."""
    api_key: str
    key: str
    translation_key: str
    device_class: BinarySensorDeviceClass
    entity_category: Optional[EntityCategory]

SECURITY_BINARY_SENSORS: List[SynologyDSMBinarySensorEntityDescription] = [
    SynologyDSMBinarySensorEntityDescription(api_key=SynoCoreSecurity.API_KEY, key='status', translation_key='status', device_class=BinarySensorDeviceClass.SAFETY)
]

STORAGE_DISK_BINARY_SENSORS: List[SynologyDSMBinarySensorEntityDescription] = [
    SynologyDSMBinarySensorEntityDescription(api_key=SynoStorage.API_KEY, key='disk_exceed_bad_sector_thr', translation_key='disk_exceed_bad_sector_thr', device_class=BinarySensorDeviceClass.SAFETY, entity_category=EntityCategory.DIAGNOSTIC),
    SynologyDSMBinarySensorEntityDescription(api_key=SynoStorage.API_KEY, key='disk_below_remain_life_thr', translation_key='disk_below_remain_life_thr', device_class=BinarySensorDeviceClass.SAFETY, entity_category=EntityCategory.DIAGNOSTIC)
]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
    """Set up the Synology NAS binary sensor."""
    data: Dict = hass.data[DOMAIN][entry.unique_id]
    api: SynoApi = data.api
    coordinator: SynologyDSMCentralUpdateCoordinator = data.coordinator_central
    assert api.storage is not None
    entities: List[SynoDSMBinarySensor] = [SynoDSMSecurityBinarySensor(api, coordinator, description) for description in SECURITY_BINARY_SENSORS]
    if api.storage.disks_ids:
        entities.extend([SynoDSMStorageBinarySensor(api, coordinator, description, disk) for disk in entry.data.get(CONF_DISKS, api.storage.disks_ids) for description in STORAGE_DISK_BINARY_SENSORS])
    async_add_entities(entities)

class SynoDSMBinarySensor(SynologyDSMBaseEntity[SynologyDSMCentralUpdateCoordinator], BinarySensorEntity):
    """Mixin for binary sensor specific attributes."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMBinarySensorEntityDescription):
        """Initialize the Synology DSM binary_sensor entity."""
        super().__init__(api, coordinator, description)

class SynoDSMSecurityBinarySensor(SynoDSMBinarySensor):
    """Representation a Synology Security binary sensor."""

    @property
    def is_on(self) -> bool:
        """Return the state."""
        return getattr(self._api.security, self.entity_description.key) != 'safe'

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return bool(self._api.security) and super().available

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return security checks details."""
        assert self._api.security is not None
        return self._api.security.status_by_check

class SynoDSMStorageBinarySensor(SynologyDSMDeviceEntity, SynoDSMBinarySensor):
    """Representation a Synology Storage binary sensor."""

    def __init__(self, api: SynoApi, coordinator: SynologyDSMCentralUpdateCoordinator, description: SynologyDSMBinarySensorEntityDescription, device_id: Optional[str] = None):
        """Initialize the Synology DSM storage binary_sensor entity."""
        super().__init__(api, coordinator, description, device_id)

    @property
    def is_on(self) -> bool:
        """Return the state."""
        return bool(getattr(self._api.storage, self.entity_description.key)(self._device_id))
