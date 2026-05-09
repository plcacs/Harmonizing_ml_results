"""Tests for the Broadlink integration."""
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from homeassistant.components.broadlink.const import DOMAIN
from homeassistant.core import HomeAssistant
from tests.common import MockConfigEntry

BROADLINK_DEVICES: dict[str, tuple[str, str, str, str, str, int, int, int]] = {
    'Entrance': ('192.168.0.11', '34ea34befc25', 'RM mini 3', 'Broadlink', 'RMMINI', 10039, 57, 8),
    'Living Room': ('192.168.0.12', '34ea34b43b5a', 'RM mini 3', 'Broadlink', 'RMMINIB', 24374, 44017, 10),
    'Office': ('192.168.0.13', '34ea34b43d22', 'RM pro', 'Broadlink', 'RMPRO', 10119, 20025, 7),
    'Garage': ('192.168.0.14', '34ea34c43f31', 'RM4 pro', 'Broadlink', 'RM4PRO', 24614, 52, 4),
    'Bedroom': ('192.168.0.15', '34ea34b45d2c', 'e-Sensor', 'Broadlink', 'A1', 10004, 20025, 5),
    'Dining room': ('192.168.0.16', '34ea34b4fd1c', 'SCB1E', 'Broadlink', 'SP4B', 20757, 57, 5),
    'Kitchen': ('192.168.0.64', '34ea34b61d2c', 'SB800TD', 'Broadlink', 'SB800TD', 20558, 57, 5),
    'Gaming room': ('192.168.0.65', '34ea34b61d2d', 'MP1-1K4S', 'Broadlink', 'MP1', 20149, 57, 5),
    'Guest room': ('192.168.0.66', '34ea34b61d2e', 'HY02/HY03', 'Hysen', 'HYS', 20141, 10024, 5),
}

@dataclass
class MockSetup:
    """Representation of a mock setup."""

class BroadlinkDevice:
    """Representation of a Broadlink device."""

    def __init__(self, name: str, host: str, mac: str, model: str, manufacturer: str, type_: str, devtype: str, fwversion: int, timeout: int) -> None:
        """Initialize the device."""
        self.name: str = name
        self.host: str = host
        self.mac: str = mac
        self.model: str = model
        self.manufacturer: str = manufacturer
        self.type: str = type_
        self.devtype: str = devtype
        self.timeout: int = timeout
        self.fwversion: int = fwversion

    async def setup_entry(self, hass: HomeAssistant, mock_api: MagicMock | None = None, mock_entry: MockConfigEntry | None = None) -> MockSetup:
        """Set up the device."""
        mock_api = mock_api or self.get_mock_api()
        mock_entry = mock_entry or self.get_mock_entry()
        mock_entry.add_to_hass(hass)
        with patch('homeassistant.components.broadlink.device.blk.gendevice', return_value=mock_api) as mock_factory:
            await hass.config_entries.async_setup(mock_entry.entry_id)
            await hass.async_block_till_done()
        return MockSetup(mock_api, mock_entry, mock_factory)

    def get_mock_api(self) -> MagicMock:
        """Return a mock device (API)."""
        mock_api = MagicMock()
        mock_api.name = self.name
        mock_api.host = (self.host, 80)
        mock_api.mac = bytes.fromhex(self.mac)
        mock_api.model = self.model
        mock_api.manufacturer = self.manufacturer
        mock_api.type = self.type
        mock_api.devtype = self.devtype
        mock_api.timeout = self.timeout
        mock_api.is_locked = False
        mock_api.auth.return_value = True
        mock_api.get_fwversion.return_value = self.fwversion
        return mock_api

    def get_mock_entry(self) -> MockConfigEntry:
        """Return a mock config entry."""
        return MockConfigEntry(domain=DOMAIN, unique_id=self.mac, title=self.name, data=self.get_entry_data())

    def get_entry_data(self) -> dict[str, str]:
        """Return entry data."""
        return {'host': self.host, 'mac': self.mac, 'type': self.devtype, 'timeout': self.timeout}

class BroadlinkMP1BG1Device(BroadlinkDevice):
    """Mock device for MP1 and BG1 with special mocking of api return values."""

    def get_mock_api(self) -> MagicMock:
        """Return a mock device (API) with support for check_power calls."""
        mock_api = super().get_mock_api()
        mock_api.check_power.return_value = {'s1': 0, 's2': 0, 's3': 0, 's4': 0}
        return mock_api

class BroadlinkSP4BDevice(BroadlinkDevice):
    """Mock device for SP4b with special mocking of api return values."""

    def get_mock_api(self) -> MagicMock:
        """Return a mock device (API) with support for get_state calls."""
        mock_api = super().get_mock_api()
        mock_api.get_state.return_value = {'pwr': 0}
        return mock_api

def get_device(name: str) -> BroadlinkDevice:
    """Get a device by name."""
    dev_type = BROADLINK_DEVICES[name][5]
    if dev_type in {20149}:
        return BroadlinkMP1BG1Device(name, *BROADLINK_DEVICES[name])
    if dev_type in {20757}:
        return BroadlinkSP4BDevice(name, *BROADLINK_DEVICES[name])
    return BroadlinkDevice(name, *BROADLINK_DEVICES[name])
