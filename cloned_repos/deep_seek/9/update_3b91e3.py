"""Support for TPLink Omada device firmware updates."""
from __future__ import annotations
from datetime import timedelta
from typing import Any, NamedTuple, cast
from tplink_omada_client import OmadaSiteClient
from tplink_omada_client.devices import OmadaFirmwareUpdate, OmadaListDevice
from tplink_omada_client.exceptions import OmadaClientException, RequestFailed
from homeassistant.components.update import UpdateDeviceClass, UpdateEntity, UpdateEntityFeature
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from . import OmadaConfigEntry
from .coordinator import POLL_DEVICES, OmadaCoordinator, OmadaDevicesCoordinator
from .entity import OmadaDeviceEntity

POLL_DELAY_UPGRADE: int = 60

class FirmwareUpdateStatus(NamedTuple):
    """Firmware update information for Omada SDN devices."""
    device: OmadaListDevice
    firmware: OmadaFirmwareUpdate | None

class OmadaFirmwareUpdateCoordinator(OmadaCoordinator[FirmwareUpdateStatus]):
    """Coordinator for getting details about available firmware updates for Omada devices."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: OmadaConfigEntry,
        omada_client: OmadaSiteClient,
        devices_coordinator: OmadaDevicesCoordinator,
    ) -> None:
        """Initialize my coordinator."""
        super().__init__(hass, config_entry, omada_client, 'Firmware Updates', poll_delay=None)
        self._devices_coordinator: OmadaDevicesCoordinator = devices_coordinator
        self._config_entry: OmadaConfigEntry = config_entry
        config_entry.async_on_unload(devices_coordinator.async_add_listener(self._handle_devices_update))

    async def _get_firmware_updates(self) -> list[FirmwareUpdateStatus]:
        devices: list[OmadaListDevice] = list(self._devices_coordinator.data.values())
        updates: list[FirmwareUpdateStatus] = [
            FirmwareUpdateStatus(
                device=d,
                firmware=None if not d.need_upgrade else await self.omada_client.get_firmware_details(d)
            ) for d in devices
        ]
        self._devices_coordinator.update_interval = timedelta(
            seconds=POLL_DELAY_UPGRADE if any((u.device.fw_download for u in updates)) else POLL_DEVICES
        )
        return updates

    async def poll_update(self) -> dict[str, FirmwareUpdateStatus]:
        """Poll the state of Omada Devices firmware update availability."""
        return {d.device.mac: d for d in await self._get_firmware_updates()}

    @callback
    def _handle_devices_update(self) -> None:
        """Handle updated data from the devices coordinator."""
        self._config_entry.async_create_background_task(
            self.hass,
            self.async_request_refresh(),
            'Omada Firmware Update Refresh'
        )

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OmadaConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up switches."""
    controller = config_entry.runtime_data
    devices: dict[str, OmadaListDevice] = controller.devices_coordinator.data
    coordinator: OmadaFirmwareUpdateCoordinator = OmadaFirmwareUpdateCoordinator(
        hass, config_entry, controller.omada_client, controller.devices_coordinator
    )
    async_add_entities((OmadaDeviceUpdate(coordinator, device) for device in devices.values()))
    await coordinator.async_request_refresh()

class OmadaDeviceUpdate(OmadaDeviceEntity[OmadaFirmwareUpdateCoordinator], UpdateEntity):
    """Firmware update status for Omada SDN devices."""
    _attr_supported_features: int = (
        UpdateEntityFeature.INSTALL |
        UpdateEntityFeature.PROGRESS |
        UpdateEntityFeature.RELEASE_NOTES
    )
    _attr_device_class: UpdateDeviceClass = UpdateDeviceClass.FIRMWARE

    def __init__(self, coordinator: OmadaFirmwareUpdateCoordinator, device: OmadaListDevice) -> None:
        """Initialize the update entity."""
        super().__init__(coordinator, device)
        self._mac: str = device.mac
        self._omada_client: OmadaSiteClient = coordinator.omada_client
        self._attr_unique_id: str = f'{device.mac}_firmware'

    @property
    def release_notes(self) -> str | None:
        """Get the release notes for the latest update."""
        status: FirmwareUpdateStatus = self.coordinator.data[self._mac]
        if status.firmware:
            return status.firmware.release_notes
        return None

    async def async_install(self, version: str | None, backup: bool, **kwargs: Any) -> None:
        """Install a firmware update."""
        try:
            await self._omada_client.start_firmware_upgrade(self.coordinator.data[self._mac].device)
        except RequestFailed as ex:
            raise HomeAssistantError('Firmware update request rejected') from ex
        except OmadaClientException as ex:
            raise HomeAssistantError('Unable to send Firmware update request. Check the controller is online.') from ex
        finally:
            await self.coordinator.async_request_refresh()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        status: FirmwareUpdateStatus = self.coordinator.data[self._mac]
        if status.firmware and status.device.need_upgrade:
            self._attr_installed_version: str = status.firmware.current_version
            self._attr_latest_version: str = status.firmware.latest_version
        else:
            self._attr_installed_version: str = status.device.firmware_version
            self._attr_latest_version: str = status.device.firmware_version
        self._attr_in_progress: bool = bool(status.device.fw_download)
        self.async_write_ha_state()
