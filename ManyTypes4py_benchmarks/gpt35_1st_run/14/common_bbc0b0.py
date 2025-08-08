from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType, GPSType

def async_see(hass: HomeAssistant, mac: str = None, dev_id: str = None, host_name: str = None, location_name: str = None, gps: GPSType = None, gps_accuracy: int = None, battery: int = None, attributes: dict = None) -> None:

class MockScannerEntity(ScannerEntity):

    @property
    def source_type(self) -> SourceType:

    @property
    def battery_level(self) -> int:

    @property
    def ip_address(self) -> str:

    @property
    def mac_address(self) -> str:

    @property
    def hostname(self) -> str:

    @property
    def is_connected(self) -> bool:

    def set_connected(self) -> None:

class MockScanner(DeviceScanner):

    def __init__(self) -> None:

    def come_home(self, device) -> None:

    def leave_home(self, device) -> None:

    def reset(self) -> None:

    def scan_devices(self) -> list:

    def get_device_name(self, device) -> str:

def mock_legacy_device_tracker_setup(hass: HomeAssistant, legacy_device_scanner) -> None:
