from typing import Any

# === Third-party dependency: habluetooth ===
# Used symbols: BaseHaRemoteScanner, BluetoothManager

# === Internal dependency: homeassistant.components.bluetooth.match ===
CALLBACK: Final
ADDRESS: Final
CONNECTABLE: Final
class BluetoothCallbackMatcherWithCallback(_BluetoothCallbackMatcherWithCallback, BluetoothCallbackMatcher):
    ...
class BluetoothCallbackMatcherIndex(BluetoothMatcherIndexBase[BluetoothCallbackMatcherWithCallback]):
    def __init__(self) -> None: ...
def ble_device_matches(matcher: BluetoothMatcherOptional, service_info: BluetoothServiceInfoBleak) -> bool: ...

# === Internal dependency: homeassistant.components.bluetooth.models ===
# re-export: from home_assistant_bluetooth import BluetoothServiceInfoBleak
BluetoothChange: Enum

# === Internal dependency: homeassistant.components.bluetooth.util ===
def async_load_history_from_system(adapters: BluetoothAdapters, storage: BluetoothStorage) -> tuple[dict[str, BluetoothServiceInfoBleak], dict[str, BluetoothServiceInfoBleak]]: ...

# === Internal dependency: homeassistant.config_entries ===
SOURCE_BLUETOOTH: str

# === Internal dependency: homeassistant.const ===
EVENT_HOMEASSISTANT_STOP: EventType[NoEventData]
EVENT_LOGGING_CHANGED: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.helpers.discovery_flow ===
def async_create_flow(hass: HomeAssistant, domain: str, context: dict[str, Any], data: Any) -> None: ...