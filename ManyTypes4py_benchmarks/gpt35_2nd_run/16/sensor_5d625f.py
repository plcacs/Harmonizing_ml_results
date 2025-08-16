from __future__ import annotations
import logging
from types import MappingProxyType
from typing import Any, Optional, Dict, List, Union, Callable
from motioneye_client.client import MotionEyeClient
from motioneye_client.const import KEY_ACTIONS
from homeassistant.components.sensor import SensorEntity, SensorEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from . import get_camera_from_cameras, listen_for_new_cameras
from .const import CONF_CLIENT, CONF_COORDINATOR, DOMAIN, TYPE_MOTIONEYE_ACTION_SENSOR
from .entity import MotionEyeEntity
_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    entry_data: Dict[str, Any] = hass.data[DOMAIN][entry.entry_id]

    @callback
    def camera_add(camera: Dict[str, Any]) -> None:
        async_add_entities([MotionEyeActionSensor(entry.entry_id, camera, entry_data[CONF_CLIENT], entry_data[CONF_COORDINATOR], entry.options)])
    listen_for_new_cameras(hass, entry, camera_add)

class MotionEyeActionSensor(MotionEyeEntity, SensorEntity):
    """motionEye action sensor camera."""
    _attr_translation_key: str = 'actions'

    def __init__(self, config_entry_id: str, camera: Dict[str, Any], client: MotionEyeClient, coordinator: DataUpdateCoordinator, options: Dict[str, Any]) -> None:
        super().__init__(config_entry_id, TYPE_MOTIONEYE_ACTION_SENSOR, camera, client, coordinator, options, SensorEntityDescription(key=TYPE_MOTIONEYE_ACTION_SENSOR, entity_registry_enabled_default=False))

    @property
    def native_value(self) -> int:
        return len(self._camera.get(KEY_ACTIONS, [])) if self._camera else 0

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Union[List[str], None]]]:
        if (actions := (self._camera.get(KEY_ACTIONS) if self._camera else None)):
            return {KEY_ACTIONS: actions}
        return None

    @callback
    def _handle_coordinator_update(self) -> None:
        self._camera = get_camera_from_cameras(self._camera_id, self.coordinator.data)
        super()._handle_coordinator_update()
