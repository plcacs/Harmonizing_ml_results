from __future__ import annotations
from typing import Any, Optional
from pyezviz import HTTPError, PyEzvizError
from homeassistant.config_entries import ConfigEntry
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
    key='version', device_class=UpdateDeviceClass.FIRMWARE
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up EZVIZ sensors based on a config entry."""
    coordinator: EzvizDataUpdateCoordinator = entry.runtime_data
    async_add_entities(
        (
            EzvizUpdateEntity(coordinator, camera, sensor, UPDATE_ENTITY_TYPES)
            for camera in coordinator.data
            for sensor, value in coordinator.data[camera].items()
            if sensor in UPDATE_ENTITY_TYPES.key if value
        )
    )


class EzvizUpdateEntity(EzvizEntity, UpdateEntity):
    """Representation of a EZVIZ Update entity."""

    _attr_supported_features: int = (
        UpdateEntityFeature.INSTALL | UpdateEntityFeature.PROGRESS | UpdateEntityFeature.RELEASE_NOTES
    )

    def __init__(
        self,
        coordinator: EzvizDataUpdateCoordinator,
        serial: str,
        sensor: str,
        description: UpdateEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, serial)
        self._attr_unique_id = f"{serial}_{sensor}"
        self.entity_description = description

    @property
    def installed_version(self) -> str:
        """Version installed and in use."""
        return self.data['version']

    @property
    def in_progress(self) -> bool:
        """Update installation progress."""
        return bool(self.data['upgrade_in_progress'])

    @property
    def latest_version(self) -> str:
        """Latest version available for install."""
        if self.data['upgrade_available']:
            return self.data['latest_firmware_info']['version']
        return self.installed_version

    def release_notes(self) -> Optional[str]:
        """Return full release notes."""
        if self.data['latest_firmware_info']:
            return self.data['latest_firmware_info'].get('desc')
        return None

    @property
    def update_percentage(self) -> Optional[int]:
        """Update installation progress."""
        if self.data['upgrade_in_progress']:
            return self.data['upgrade_percent']
        return None

    async def async_install(self, version: str, backup: bool, **kwargs: Any) -> None:
        """Install an update."""
        try:
            await self.hass.async_add_executor_job(
                self.coordinator.ezviz_client.upgrade_device, self._serial
            )
        except (HTTPError, PyEzvizError) as err:
            raise HomeAssistantError(f"Failed to update firmware on {self.name}") from err