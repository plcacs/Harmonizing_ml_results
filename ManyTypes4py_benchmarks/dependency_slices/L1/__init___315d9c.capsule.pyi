# === Third-party dependency: bleak ===
class BleakClient:
    def address(self) -> str: ...
    def mtu_size(self) -> int: ...
    def is_connected(self) -> bool: ...
    def services(self) -> BleakGATTServiceCollection: ...

# === Third-party dependency: bleak.backends.scanner ===
class AdvertisementData(NamedTuple): ...
# re-export: from .device import BLEDevice

# === Third-party dependency: bluetooth_adapters ===
# Used symbols: DEFAULT_ADDRESS

# === Third-party dependency: habluetooth ===
# Used symbols: BaseHaScanner, BluetoothManager, get_manager

# === Internal dependency: homeassistant.components.bluetooth ===
from home_assistant_bluetooth import BluetoothServiceInfoBleak
from .api import async_get_advertisement_callback
from .const import DOMAIN
from .const import SOURCE_LOCAL

# === Internal dependency: homeassistant.setup ===
async def async_setup_component(hass, domain, config): ...

# === Internal dependency: tests.common ===
class MockConfigEntry(config_entries.ConfigEntry):
    def __init__(self, *, data=..., disabled_by=..., domain=..., entry_id=..., minor_version=..., options=..., pref_disable_new_entities=..., pref_disable_polling=..., reason=..., source=..., state=..., title=..., unique_id=..., version=...): ...
    def add_to_hass(self, hass): ...