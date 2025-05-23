"""Connected Wi-Fi device scanners for TP-Link Omada access points."""
import logging
from tplink_omada_client.clients import OmadaWirelessClient
from homeassistant.components.device_tracker import ScannerEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from . import OmadaConfigEntry
from .config_flow import CONF_SITE
from .controller import OmadaClientsCoordinator
_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up device trackers and scanners."""
    controller = config_entry.runtime_data
    site_id = config_entry.data[CONF_SITE]
    async_add_entities([OmadaClientScannerEntity(site_id, client.mac, client.name, controller.clients_coordinator) async for client in controller.omada_client.get_known_clients() if isinstance(client, OmadaWirelessClient)])

class OmadaClientScannerEntity(CoordinatorEntity[OmadaClientsCoordinator], ScannerEntity):
    """Entity for a client connected to the Omada network."""
    _client_details = None

    def __init__(self, site_id, client_id, display_name, coordinator):
        """Initialize the scanner."""
        super().__init__(coordinator)
        self._site_id = site_id
        self._client_id = client_id
        self._attr_name = display_name

    def _do_update(self):
        self._client_details = self.coordinator.data.get(self._client_id)

    async def async_added_to_hass(self):
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self._do_update()

    @callback
    def _handle_coordinator_update(self):
        """Handle updated data from the coordinator."""
        self._do_update()
        self.async_write_ha_state()

    @property
    def ip_address(self):
        """Return the primary ip address of the device."""
        return self._client_details.ip if self._client_details else None

    @property
    def mac_address(self):
        """Return the mac address of the device."""
        return self._client_id

    @property
    def hostname(self):
        """Return hostname of the device."""
        return self._client_details.host_name if self._client_details else None

    @property
    def is_connected(self):
        """Return true if the device is connected to the network."""
        return self._client_details.is_active if self._client_details else False

    @property
    def unique_id(self):
        """Return the unique id of the device."""
        return f'scanner_{self._site_id}_{self._client_id}'