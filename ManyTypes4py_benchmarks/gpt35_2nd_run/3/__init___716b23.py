from dataclasses import dataclass
from typing import Any
from homeassistant.core import HomeAssistant

@dataclass
class MockSetup:
    """Representation of a mock setup."""
    mock_api: Any
    mock_entry: Any
    mock_factory: Any

class BroadlinkDevice:
    """Representation of a Broadlink device."""

    def __init__(self, name: str, host: str, mac: str, model: str, manufacturer: str, type_: str, devtype: int, fwversion: int, timeout: int):
        """Initialize the device."""
        self.name: str = name
        self.host: str = host
        self.mac: str = mac
        self.model: str = model
        self.manufacturer: str = manufacturer
        self.type: str = type_
        self.devtype: int = devtype
        self.timeout: int = timeout
        self.fwversion: int = fwversion

    async def setup_entry(self, hass: HomeAssistant, mock_api: Any = None, mock_entry: Any = None) -> MockSetup:
        """Set up the device."""
        mock_api: Any = mock_api or self.get_mock_api()
        mock_entry: Any = mock_entry or self.get_mock_entry()
        mock_entry.add_to_hass(hass)
        with patch('homeassistant.components.broadlink.device.blk.gendevice', return_value=mock_api) as mock_factory:
            await hass.config_entries.async_setup(mock_entry.entry_id)
            await hass.async_block_till_done()
        return MockSetup(mock_api, mock_entry, mock_factory)

    def get_mock_api(self) -> Any:
        """Return a mock device (API)."""
        mock_api: Any = MagicMock()
        mock_api.name: str = self.name
        mock_api.host: tuple = (self.host, 80)
        mock_api.mac: bytes = bytes.fromhex(self.mac)
        mock_api.model: str = self.model
        mock_api.manufacturer: str = self.manufacturer
        mock_api.type: str = self.type
        mock_api.devtype: int = self.devtype
        mock_api.timeout: int = self.timeout
        mock_api.is_locked: bool = False
        mock_api.auth.return_value: bool = True
        mock_api.get_fwversion.return_value: int = self.fwversion
        return mock_api

    def get_mock_entry(self) -> Any:
        """Return a mock config entry."""
        return MockConfigEntry(domain=DOMAIN, unique_id=self.mac, title=self.name, data=self.get_entry_data())

    def get_entry_data(self) -> dict:
        """Return entry data."""
        return {'host': self.host, 'mac': self.mac, 'type': self.devtype, 'timeout': self.timeout}

class BroadlinkMP1BG1Device(BroadlinkDevice):
    """Mock device for MP1 and BG1 with special mocking of api return values."""

    def get_mock_api(self) -> Any:
        """Return a mock device (API) with support for check_power calls."""
        mock_api: Any = super().get_mock_api()
        mock_api.check_power.return_value: dict = {'s1': 0, 's2': 0, 's3': 0, 's4': 0}
        return mock_api

class BroadlinkSP4BDevice(BroadlinkDevice):
    """Mock device for SP4b with special mocking of api return values."""

    def get_mock_api(self) -> Any:
        """Return a mock device (API) with support for get_state calls."""
        mock_api: Any = super().get_mock_api()
        mock_api.get_state.return_value: dict = {'pwr': 0}
        return mock_api

def get_device(name: str) -> BroadlinkDevice:
    """Get a device by name."""
    dev_type: int = BROADLINK_DEVICES[name][5]
    if dev_type in {20149}:
        return BroadlinkMP1BG1Device(name, *BROADLINK_DEVICES[name])
    if dev_type in {20757}:
        return BroadlinkSP4BDevice(name, *BROADLINK_DEVICES[name])
    return BroadlinkDevice(name, *BROADLINK_DEVICES[name])
