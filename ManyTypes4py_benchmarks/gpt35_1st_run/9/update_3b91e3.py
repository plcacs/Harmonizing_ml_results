from __future__ import annotations
from datetime import timedelta
from typing import Any, NamedTuple
from tplink_omada_client import OmadaSiteClient
from tplink_omada_client.devices import OmadaFirmwareUpdate, OmadaListDevice
from tplink_omada_client.exceptions import OmadaClientException, RequestFailed
from homeassistant.components.update import UpdateDeviceClass, UpdateEntity, UpdateEntityFeature
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import OmadaConfigEntry
from .coordinator import POLL_DEVICES, OmadaCoordinator, OmadaDevicesCoordinator
from .entity import OmadaDeviceEntity

POLL_DELAY_UPGRADE: int = 60

class FirmwareUpdateStatus(NamedTuple):
    device: OmadaListDevice
    firmware: OmadaFirmwareUpdate

class OmadaFirmwareUpdateCoordinator(OmadaCoordinator[FirmwareUpdateStatus]):
    def __init__(self, hass: HomeAssistant, config_entry: OmadaConfigEntry, omada_client: OmadaSiteClient, devices_coordinator: OmadaDevicesCoordinator):
        ...

    async def _get_firmware_updates(self) -> list[FirmwareUpdateStatus]:
        ...

    async def poll_update(self) -> dict[str, FirmwareUpdateStatus]:
        ...

    @callback
    def _handle_devices_update(self):
        ...

async def async_setup_entry(hass: HomeAssistant, config_entry: OmadaConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
    ...

class OmadaDeviceUpdate(OmadaDeviceEntity[OmadaFirmwareUpdateCoordinator], UpdateEntity):
    _attr_supported_features: int = UpdateEntityFeature.INSTALL | UpdateEntityFeature.PROGRESS | UpdateEntityFeature.RELEASE_NOTES
    _attr_device_class: str = UpdateDeviceClass.FIRMWARE

    def __init__(self, coordinator: OmadaFirmwareUpdateCoordinator, device: OmadaListDevice):
        ...

    def release_notes(self) -> str:
        ...

    async def async_install(self, version: str, backup: Any, **kwargs: Any):
        ...

    @callback
    def _handle_coordinator_update(self):
        ...
