"""The motionEye integration."""
from __future__ import annotations
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple
from motioneye_client.client import MotionEyeClient
from motioneye_client.const import KEY_ID
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityDescription
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator
from . import get_motioneye_device_identifier

def get_motioneye_entity_unique_id(config_entry_id: str, camera_id: int, entity_type: str) -> str:
    """Get the unique_id for a motionEye entity."""
    return f'{config_entry_id}_{camera_id}_{entity_type}'

class MotionEyeEntity(CoordinatorEntity):
    """Base class for motionEye entities."""
    _attr_has_entity_name: bool = True

    def __init__(
        self, 
        config_entry_id: str, 
        type_name: str, 
        camera: Dict[str, Any], 
        client: MotionEyeClient, 
        coordinator: DataUpdateCoordinator, 
        options: Dict[str, Any], 
        entity_description: Optional[EntityDescription] = None
    ) -> None:
        """Initialize a motionEye entity."""
        self._camera_id: int = camera[KEY_ID]
        self._device_identifier: Tuple[str, str] = get_motioneye_device_identifier(config_entry_id, self._camera_id)
        self._unique_id: str = get_motioneye_entity_unique_id(config_entry_id, self._camera_id, type_name)
        self._client: MotionEyeClient = client
        self._camera: Dict[str, Any] = camera
        self._options: Dict[str, Any] = options
        if entity_description is not None:
            self.entity_description = entity_description
        super().__init__(coordinator)

    @property
    def unique_id(self) -> str:
        """Return a unique id for this instance."""
        return self._unique_id

    @property
    def device_info(self) -> DeviceInfo:
        """Return the device information."""
        return DeviceInfo(identifiers={self._device_identifier})

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self._camera is not None and super().available
