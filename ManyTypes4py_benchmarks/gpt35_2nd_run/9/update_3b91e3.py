from __future__ import annotations
from datetime import timedelta
from typing import Any, NamedTuple, Dict, List, Union, Optional
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
    firmware: Optional[OmadaFirmwareUpdate]

class OmadaFirmwareUpdateCoordinator(OmadaCoordinator[FirmwareUpdateStatus]):
    def __init__(self, hass: HomeAssistant, config_entry: OmadaConfigEntry, omada_client: OmadaSiteClient, devices_coordinator: OmadaDevicesCoordinator):
        super().__init__(hass, config_entry, omada_client, 'Firmware Updates', poll_delay=None)
        self._devices_coordinator = devices_coordinator
        self._config_entry = config_entry
        config_entry.async_on_unload(devices_coordinator.async_add_listener(self._handle_devices_update))

    async def _get_firmware_updates(self) -> List[FirmwareUpdateStatus]:
        devices = list(self._devices_coordinator.data.values())
        updates = [FirmwareUpdateStatus(device=d, firmware=None if not d.need_upgrade else await self.omada_client.get_firmware_details(d)) for d in devices]
        self._devices_coordinator.update_interval = timedelta(seconds=POLL_DELAY_UPGRADE if any((u.device.fw_download for u in updates)) else POLL_DEVICES)
        return updates

    async def poll_update(self) -> Dict[str, FirmwareUpdateStatus]:
        return {d.device.mac: d for d in await self._get_firmware_updates()}

    @callback
    def _handle_devices_update(self):
        self._config_entry.async_create_background_task(self.hass, self.async_request_refresh(), 'Omada Firmware Update Refresh')

async def async_setup_entry(hass: HomeAssistant, config_entry: OmadaConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
    controller = config_entry.runtime_data
    devices = controller.devices_coordinator.data
    coordinator = OmadaFirmwareUpdateCoordinator(hass, config_entry, controller.omada_client, controller.devices_coordinator)
    async_add_entities((OmadaDeviceUpdate(coordinator, device) for device in devices.values()))
    await coordinator.async_request_refresh()

class OmadaDeviceUpdate(OmadaDeviceEntity[OmadaFirmwareUpdateCoordinator], UpdateEntity):
    _attr_supported_features: int = UpdateEntityFeature.INSTALL | UpdateEntityFeature.PROGRESS | UpdateEntityFeature.RELEASE_NOTES
    _attr_device_class: str = UpdateDeviceClass.FIRMWARE

    def __init__(self, coordinator: OmadaFirmwareUpdateCoordinator, device: OmadaListDevice):
        super().__init__(coordinator, device)
        self._mac: str = device.mac
        self._omada_client: OmadaSiteClient = coordinator.omada_client
        self._attr_unique_id: str = f'{device.mac}_firmware'

    def release_notes(self) -> Optional[str]:
        status = self.coordinator.data[self._mac]
        if status.firmware:
            return status.firmware.release_notes
        return None

    async def async_install(self, version: str, backup: Any, **kwargs: Any):
        try:
            await self._omada_client.start_firmware_upgrade(self.coordinator.data[self._mac].device)
        except RequestFailed as ex:
            raise HomeAssistantError('Firmware update request rejected') from ex
        except OmadaClientException as ex:
            raise HomeAssistantError('Unable to send Firmware update request. Check the controller is online.') from ex
        finally:
            await self.coordinator.async_request_refresh()

    @callback
    def _handle_coordinator_update(self):
        status = self.coordinator.data[self._mac]
        if status.firmware and status.device.need_upgrade:
            self._attr_installed_version = status.firmware.current_version
            self._attr_latest_version = status.firmware.latest_version
        else:
            self._attr_installed_version = status.device.firmware_version
            self._attr_latest_version = status.device.firmware_version
        self._attr_in_progress = status.device.fw_download
        self.async_write_ha_state()
