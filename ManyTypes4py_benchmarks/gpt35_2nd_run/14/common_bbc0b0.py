from typing import Optional, Dict, Any, List

def async_see(hass: HomeAssistant, mac: Optional[str] = None, dev_id: Optional[str] = None, host_name: Optional[str] = None, location_name: Optional[str] = None, gps: Optional[GPSType] = None, gps_accuracy: Optional[int] = None, battery: Optional[int] = None, attributes: Optional[Dict[str, Any]] = None) -> None:

class MockScannerEntity(ScannerEntity):
    def __init__(self) -> None:
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
    def scan_devices(self) -> List[str]:
    def get_device_name(self, device) -> Optional[str]:

def mock_legacy_device_tracker_setup(hass: HomeAssistant, legacy_device_scanner: MockScanner) -> None:
    async def _async_get_scanner(hass: HomeAssistant, config: ConfigType) -> MockScanner:
