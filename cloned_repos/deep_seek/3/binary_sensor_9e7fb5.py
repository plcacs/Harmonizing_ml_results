"""Component providing binary sensors for UniFi Protect."""
from __future__ import annotations
from collections.abc import Sequence, Callable
from typing import Any, TypeVar, Generic, Optional, Union, cast
import dataclasses
from uiprotect.data import NVR, Camera, ModelType, MountType, ProtectAdoptableDeviceModel, Sensor, SmartDetectObjectType
from uiprotect.data.nvr import UOSDisk
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity, BinarySensorEntityDescription
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback, AddConfigEntryEntitiesCallback
from .data import ProtectData, ProtectDeviceType, UFPConfigEntry
from .entity import BaseProtectEntity, EventEntityMixin, PermRequired, ProtectDeviceEntity, ProtectEntityDescription, ProtectEventMixin, ProtectIsOnEntity, ProtectNVREntity, async_all_device_entities

T = TypeVar('T', bound=ProtectAdoptableDeviceModel)
U = TypeVar('U', bound=Union[NVR, ProtectAdoptableDeviceModel])

_KEY_DOOR = 'door'

@dataclasses.dataclass(frozen=True, kw_only=True)
class ProtectBinaryEntityDescription(ProtectEntityDescription, BinarySensorEntityDescription):
    """Describes UniFi Protect Binary Sensor entity."""

@dataclasses.dataclass(frozen=True, kw_only=True)
class ProtectBinaryEventEntityDescription(ProtectEventMixin, BinarySensorEntityDescription):
    """Describes UniFi Protect Binary Sensor entity."""

MOUNT_DEVICE_CLASS_MAP: dict[MountType, BinarySensorDeviceClass] = {
    MountType.GARAGE: BinarySensorDeviceClass.GARAGE_DOOR,
    MountType.WINDOW: BinarySensorDeviceClass.WINDOW,
    MountType.DOOR: BinarySensorDeviceClass.DOOR
}

CAMERA_SENSORS: tuple[ProtectBinaryEntityDescription, ...] = (
    ProtectBinaryEntityDescription(key='dark', name='Is dark', icon='mdi:brightness-6', ufp_value='is_dark'),
    # ... (rest of the CAMERA_SENSORS tuple remains the same)
)

LIGHT_SENSORS: tuple[ProtectBinaryEntityDescription, ...] = (
    ProtectBinaryEntityDescription(key='dark', name='Is dark', icon='mdi:brightness-6', ufp_value='is_dark'),
    # ... (rest of the LIGHT_SENSORS tuple remains the same)
)

MOUNTABLE_SENSE_SENSORS: tuple[ProtectBinaryEntityDescription, ...] = (
    ProtectBinaryEntityDescription(key=_KEY_DOOR, name='Contact', device_class=BinarySensorDeviceClass.DOOR, ufp_value='is_opened', ufp_enabled='is_contact_sensor_enabled'),
)

SENSE_SENSORS: tuple[ProtectBinaryEntityDescription, ...] = (
    ProtectBinaryEntityDescription(key='leak', name='Leak', device_class=BinarySensorDeviceClass.MOISTURE, ufp_value='is_leak_detected', ufp_enabled='is_leak_sensor_enabled'),
    # ... (rest of the SENSE_SENSORS tuple remains the same)
)

EVENT_SENSORS: tuple[ProtectBinaryEventEntityDescription, ...] = (
    ProtectBinaryEventEntityDescription(key='doorbell', name='Doorbell', device_class=BinarySensorDeviceClass.OCCUPANCY, icon='mdi:doorbell-video', ufp_required_field='feature_flags.is_doorbell', ufp_event_obj='last_ring_event'),
    # ... (rest of the EVENT_SENSORS tuple remains the same)
)

DOORLOCK_SENSORS: tuple[ProtectBinaryEntityDescription, ...] = (
    ProtectBinaryEntityDescription(key='battery_low', name='Battery low', device_class=BinarySensorDeviceClass.BATTERY, entity_category=EntityCategory.DIAGNOSTIC, ufp_value='battery_status.is_low'),
    # ... (rest of the DOORLOCK_SENSORS tuple remains the same)
)

VIEWER_SENSORS: tuple[ProtectBinaryEntityDescription, ...] = (
    ProtectBinaryEntityDescription(key='ssh', name='SSH enabled', icon='mdi:lock', entity_registry_enabled_default=False, entity_category=EntityCategory.DIAGNOSTIC, ufp_value='is_ssh_enabled', ufp_perm=PermRequired.NO_WRITE),
)

DISK_SENSORS: tuple[ProtectBinaryEntityDescription, ...] = (
    ProtectBinaryEntityDescription(key='disk_health', device_class=BinarySensorDeviceClass.PROBLEM, entity_category=EntityCategory.DIAGNOSTIC),
)

_MODEL_DESCRIPTIONS: dict[ModelType, tuple[ProtectBinaryEntityDescription, ...]] = {
    ModelType.CAMERA: CAMERA_SENSORS,
    ModelType.LIGHT: LIGHT_SENSORS,
    ModelType.SENSOR: SENSE_SENSORS,
    ModelType.DOORLOCK: DOORLOCK_SENSORS,
    ModelType.VIEWPORT: VIEWER_SENSORS
}

_MOUNTABLE_MODEL_DESCRIPTIONS: dict[ModelType, tuple[ProtectBinaryEntityDescription, ...]] = {
    ModelType.SENSOR: MOUNTABLE_SENSE_SENSORS
}

class ProtectDeviceBinarySensor(ProtectIsOnEntity, ProtectDeviceEntity, BinarySensorEntity):
    """A UniFi Protect Device Binary Sensor."""

class MountableProtectDeviceBinarySensor(ProtectDeviceBinarySensor):
    """A UniFi Protect Device Binary Sensor that can change device class at runtime."""
    _state_attrs: tuple[str, ...] = ('_attr_available', '_attr_is_on', '_attr_device_class')

    @callback
    def _async_update_device_from_protect(self, device: ProtectAdoptableDeviceModel) -> None:
        super()._async_update_device_from_protect(device)
        self._attr_device_class = MOUNT_DEVICE_CLASS_MAP.get(device.mount_type, BinarySensorDeviceClass.DOOR)

class ProtectDiskBinarySensor(ProtectNVREntity, BinarySensorEntity):
    """A UniFi Protect NVR Disk Binary Sensor."""
    _state_attrs: tuple[str, ...] = ('_attr_available', '_attr_is_on')

    def __init__(self, data: ProtectData, device: NVR, description: ProtectBinaryEntityDescription, disk: UOSDisk) -> None:
        """Initialize the Binary Sensor."""
        self._disk: UOSDisk = disk
        index: int = self._disk.slot - 1
        description = dataclasses.replace(description, key=f'{description.key}_{index}', name=f'{disk.type} {disk.slot}')
        super().__init__(data, device, description)

    @callback
    def _async_update_device_from_protect(self, device: NVR) -> None:
        super()._async_update_device_from_protect(device)
        slot: int = self._disk.slot
        self._attr_available = False
        available: bool = self.data.last_update_success
        assert device.system_info.ustorage is not None
        for disk in device.system_info.ustorage.disks:
            if disk.slot == slot:
                self._disk = disk
                self._attr_available = available
                break
        self._attr_is_on = not self._disk.is_healthy

class ProtectEventBinarySensor(EventEntityMixin, BinarySensorEntity):
    """A UniFi Protect Device Binary Sensor for events."""
    _state_attrs: tuple[str, ...] = ('_attr_available', '_attr_is_on', '_attr_extra_state_attributes')

    @callback
    def _set_event_done(self) -> None:
        self._attr_is_on = False
        self._attr_extra_state_attributes = {}

    @callback
    def _async_update_device_from_protect(self, device: Camera) -> None:
        description: ProtectBinaryEventEntityDescription = cast(ProtectBinaryEventEntityDescription, self.entity_description)
        prev_event = self._event
        prev_event_end = self._event_end
        super()._async_update_device_from_protect(device)
        if (event := description.get_event_obj(device)):
            self._event = event
            self._event_end = event.end if event else None
        if not (event and (description.ufp_obj_type is None or description.has_matching_smart(event)) and (not self._event_already_ended(prev_event, prev_event_end)):
            self._set_event_done()
            return
        self._attr_is_on = True
        self._set_event_attrs(event)
        if event.end:
            self._async_event_with_immediate_end()

MODEL_DESCRIPTIONS_WITH_CLASS: tuple[tuple[dict[ModelType, tuple[ProtectBinaryEntityDescription, ...]], type[ProtectDeviceBinarySensor]], ...] = (
    (_MODEL_DESCRIPTIONS, ProtectDeviceBinarySensor),
    (_MOUNTABLE_MODEL_DESCRIPTIONS, MountableProtectDeviceBinarySensor)
)

@callback
def _async_event_entities(data: ProtectData, ufp_device: Optional[Camera] = None) -> list[ProtectEventBinarySensor]:
    entities: list[ProtectEventBinarySensor] = []
    for device in data.get_cameras() if ufp_device is None else [ufp_device]:
        entities.extend((ProtectEventBinarySensor(data, device, description) for description in EVENT_SENSORS if description.has_required(device)))
    return entities

@callback
def _async_nvr_entities(data: ProtectData) -> list[ProtectDiskBinarySensor]:
    device: NVR = data.api.bootstrap.nvr
    if (ustorage := device.system_info.ustorage) is None:
        return []
    return [ProtectDiskBinarySensor(data, device, description, disk) for disk in ustorage.disks for description in DISK_SENSORS if disk.has_disk]

async def async_setup_entry(
    hass: HomeAssistant,
    entry: UFPConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up binary sensors for UniFi Protect integration."""
    data: ProtectData = entry.runtime_data

    @callback
    def _add_new_device(device: ProtectAdoptableDeviceModel) -> None:
        entities: list[Union[ProtectDeviceBinarySensor, ProtectEventBinarySensor]] = []
        for model_descriptions, klass in MODEL_DESCRIPTIONS_WITH_CLASS:
            entities += async_all_device_entities(data, klass, model_descriptions=model_descriptions, ufp_device=device)
        if device.is_adopted and isinstance(device, Camera):
            entities += _async_event_entities(data, ufp_device=device)
        async_add_entities(entities)
    data.async_subscribe_adopt(_add_new_device)
    entities: list[Union[ProtectDeviceBinarySensor, ProtectEventBinarySensor, ProtectDiskBinarySensor]] = []
    for model_descriptions, klass in MODEL_DESCRIPTIONS_WITH_CLASS:
        entities += async_all_device_entities(data, klass, model_descriptions=model_descriptions)
    entities += _async_event_entities(data)
    entities += _async_nvr_entities(data)
    async_add_entities(entities)
