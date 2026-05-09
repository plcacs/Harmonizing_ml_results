from __future__ import annotations
from collections.abc import Sequence
import dataclasses
from uiprotect.data import NVR, Camera, ModelType, MountType, ProtectAdoptableDeviceModel, Sensor, SmartDetectObjectType
from uiprotect.data.nvr import UOSDisk
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity, BinarySensorEntityDescription
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .data import ProtectData, ProtectDeviceType, UFPConfigEntry
from .entity import BaseProtectEntity, EventEntityMixin, PermRequired, ProtectDeviceEntity, ProtectEntityDescription, ProtectIsOnEntity, ProtectNVREntity, async_all_device_entities

class ProtectBinaryEntityDescription(ProtectEntityDescription, BinarySensorEntityDescription):
    """Describes UniFi Protect Binary Sensor entity."""
    key: str
    name: str
    icon: str
    ufp_value: str
    ufp_perm: PermRequired
    ufp_required_field: str | None
    ufp_enabled: str | None
    ufp_obj_type: SmartDetectObjectType | None

class ProtectBinaryEventEntityDescription(ProtectEventMixin, BinarySensorEntityDescription):
    """Describes UniFi Protect Binary Sensor entity."""
    key: str
    name: str
    ufp_required_field: str | None
    ufp_event_obj: str
    ufp_obj_type: SmartDetectObjectType | None

@dataclasses.dataclass(frozen=True, kw_only=True)
class ProtectDeviceBinarySensor(ProtectIsOnEntity, ProtectDeviceEntity, BinarySensorEntity):
    """A UniFi Protect Device Binary Sensor."""

class MountableProtectDeviceBinarySensor(ProtectDeviceBinarySensor):
    """A UniFi Protect Device Binary Sensor that can change device class at runtime."""
    _state_attrs: tuple[str, ...] = ('_attr_available', '_attr_is_on', '_attr_device_class')

    @callback
    def _async_update_device_from_protect(self, device):
        super()._async_update_device_from_protect(device)
        self._attr_device_class = MOUNT_DEVICE_CLASS_MAP.get(self.device.mount_type, BinarySensorDeviceClass.DOOR)

class ProtectDiskBinarySensor(ProtectNVREntity, BinarySensorEntity):
    """A UniFi Protect NVR Disk Binary Sensor."""
    _state_attrs: tuple[str, ...] = ('_attr_available', '_attr_is_on')

    def __init__(self, data: ProtectData, device: ProtectDeviceType, description: ProtectBinaryEntityDescription, disk: UOSDisk):
        """Initialize the Binary Sensor."""
        self._disk = disk
        index = self._disk.slot - 1
        description = dataclasses.replace(description, key=f'{description.key}_{index}', name=f'{disk.type} {disk.slot}')
        super().__init__(data, device, description)

    @callback
    def _async_update_device_from_protect(self, device):
        super()._async_update_device_from_protect(device)
        slot = self._disk.slot
        self._attr_available = False
        available = self.data.last_update_success
        assert self.device.system_info.ustorage is not None
        for disk in self.device.system_info.ustorage.disks:
            if disk.slot == slot:
                self._disk = disk
                self._attr_available = available
                break
        self._attr_is_on = not self._disk.is_healthy

class ProtectEventBinarySensor(EventEntityMixin, BinarySensorEntity):
    """A UniFi Protect Device Binary Sensor for events."""
    _state_attrs: tuple[str, ...] = ('_attr_available', '_attr_is_on', '_attr_extra_state_attributes')

    @callback
    def _set_event_done(self):
        self._attr_is_on = False
        self._attr_extra_state_attributes = {}

    @callback
    def _async_update_device_from_protect(self, device):
        description = self.entity_description
        prev_event = self._event
        prev_event_end = self._event_end
        super()._async_update_device_from_protect(device)
        if (event := description.get_event_obj(device)):
            self._event = event
            self._event_end = event.end if event else None
        if not (event and (description.ufp_obj_type is None or description.has_matching_smart(event)) and (not self._event_already_ended(prev_event, prev_event_end))):
            self._set_event_done()
            return
        self._attr_is_on = True
        self._set_event_attrs(event)
        if event.end:
            self._async_event_with_immediate_end()

_MODEL_DESCRIPTIONS = {ModelType.CAMERA: CAMERA_SENSORS, ModelType.LIGHT: LIGHT_SENSORS, ModelType.SENSOR: SENSE_SENSORS, ModelType.DOORLOCK: DOORLOCK_SENSORS, ModelType.VIEWPORT: VIEWER_SENSORS}

_MOUNTABLE_MODEL_DESCRIPTIONS = {ModelType.SENSOR: MOUNTABLE_SENSE_SENSORS}

@callback
def _async_event_entities(data: ProtectData, ufp_device: ProtectDeviceType | None):
    entities: list[ProtectEventBinarySensor] = []
    for device in data.get_cameras() if ufp_device is None else [ufp_device]:
        entities.extend((ProtectEventBinarySensor(data, device, description) for description in EVENT_SENSORS if description.has_required(device)))
    return entities

@callback
def _async_nvr_entities(data: ProtectData):
    device = data.api.bootstrap.nvr
    if (ustorage := device.system_info.ustorage) is None:
        return []
    return [ProtectDiskBinarySensor(data, device, description, disk) for disk in ustorag