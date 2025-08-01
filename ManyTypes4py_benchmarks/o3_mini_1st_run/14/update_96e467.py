from __future__ import annotations
from typing import Any, Dict, Optional
from pyezviz import HTTPError, PyEzvizError
from homeassistant.components.update import (
    UpdateDeviceClass,
    UpdateEntity,
    UpdateEntityDescription,
    UpdateEntityFeature,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .coordinator import EzvizConfigEntry, EzvizDataUpdateCoordinator
from .entity import EzvizEntity

PARALLEL_UPDATES: int = 1
UPDATE_ENTITY_TYPES: UpdateEntityDescription = UpdateEntityDescription(
    key="version", device_class=UpdateDeviceClass.FIRMWARE
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: EzvizConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator: EzvizDataUpdateCoordinator = entry.runtime_data
    async_add_entities(
        (
            EzvizUpdateEntity(coordinator, camera, sensor, UPDATE_ENTITY_TYPES)
            for camera in coordinator.data
            for sensor, value in coordinator.data[camera].items()
            if sensor in UPDATE_ENTITY_TYPES.key and value
        )
    )

class EzvizUpdateEntity(EzvizEntity, UpdateEntity):
    _attr_supported_features: int = UpdateEntityFeature.INSTALL | UpdateEntityFeature.PROGRESS | UpdateEntityFeature.RELEASE_NOTES

    def __init__(
        self,
        coordinator: EzvizDataUpdateCoordinator,
        serial: str,
        sensor: str,
        description: UpdateEntityDescription,
    ) -> None:
        super().__init__(coordinator, serial)
        self._attr_unique_id: str = f"{serial}_{sensor}"
        self.entity_description: UpdateEntityDescription = description

    @property
    def installed_version(self) -> str:
        return self.data["version"]

    @property
    def in_progress(self) -> bool:
        return bool(self.data["upgrade_in_progress"])

    @property
    def latest_version(self) -> str:
        if self.data["upgrade_available"]:
            return self.data["latest_firmware_info"]["version"]
        return self.installed_version

    def release_notes(self) -> Optional[str]:
        if self.data["latest_firmware_info"]:
            return self.data["latest_firmware_info"].get("desc")
        return None

    @property
    def update_percentage(self) -> Optional[int]:
        if self.data["upgrade_in_progress"]:
            return self.data["upgrade_percent"]
        return None

    async def async_install(self, version: str, backup: bool, **kwargs: Any) -> None:
        try:
            await self.hass.async_add_executor_job(
                self.coordinator.ezviz_client.upgrade_device, self._serial
            )
        except (HTTPError, PyEzvizError) as err:
            raise HomeAssistantError(f"Failed to update firmware on {self.name}") from err