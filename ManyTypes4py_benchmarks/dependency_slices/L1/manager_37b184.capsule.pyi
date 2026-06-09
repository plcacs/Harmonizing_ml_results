# === Third-party dependency: habluetooth ===
# Used symbols: BaseHaRemoteScanner, BluetoothManager

# === Internal dependency: homeassistant.components.bluetooth.match ===
class BluetoothCallbackMatcherWithCallback(_BluetoothCallbackMatcherWithCallback, BluetoothCallbackMatcher):
    ...
class BluetoothCallbackMatcherIndex(BluetoothMatcherIndexBase[BluetoothCallbackMatcherWithCallback]):
    def __init__(self): ...
def ble_device_matches(matcher, service_info): ...
CALLBACK = 'callback'
ADDRESS = 'address'
CONNECTABLE = 'connectable'

# === Internal dependency: homeassistant.components.bluetooth.models ===
from home_assistant_bluetooth import BluetoothServiceInfoBleak
BluetoothChange = Enum(...)

# === Internal dependency: homeassistant.components.bluetooth.util ===
def async_load_history_from_system(adapters, storage): ...

# === Internal dependency: homeassistant.config_entries ===
SOURCE_BLUETOOTH = 'bluetooth'

# === Internal dependency: homeassistant.const ===
EVENT_HOMEASSISTANT_STOP = EventType(...)
EVENT_LOGGING_CHANGED = 'logging_changed'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.helpers.discovery_flow ===
def async_create_flow(hass, domain, context, data): ...

# === Internal dependency: homeassistant.util.event_type ===
class EventType(Generic[_DataT]): ...