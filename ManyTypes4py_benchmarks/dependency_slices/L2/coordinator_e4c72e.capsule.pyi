from typing import Any

# === Internal dependency: homeassistant.components.bluetooth ===
# re-export: from habluetooth import BluetoothScanningMode
# re-export: from home_assistant_bluetooth import BluetoothServiceInfoBleak
# re-export: from .api import async_last_service_info
# re-export: from .api import async_register_callback
# re-export: from .api import async_track_unavailable
# re-export: from .models import BluetoothChange

# === Internal dependency: homeassistant.components.bluetooth.match ===
class BluetoothCallbackMatcher(BluetoothMatcherOptional, BluetoothCallbackMatcherOptional): ...

# === Internal dependency: homeassistant.components.ibeacon.const ===
DOMAIN: str
SIGNAL_IBEACON_DEVICE_NEW: str
SIGNAL_IBEACON_DEVICE_UNAVAILABLE: str
SIGNAL_IBEACON_DEVICE_SEEN: str
UNAVAILABLE_TIMEOUT: int
UPDATE_INTERVAL: timedelta
MAX_IDS: int
MAX_IDS_PER_UUID: int
MIN_SEEN_TRANSIENT_NEW: Any
CONF_IGNORE_ADDRESSES: str
CONF_IGNORE_UUIDS: str
CONF_ALLOW_NAMELESS_UUIDS: str

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.helpers.dispatcher ===
def async_dispatcher_send(hass: HomeAssistant, signal: SignalType[*_Ts,], *args: *_Ts) -> None: ...
def async_dispatcher_send(hass: HomeAssistant, signal: str, *args: Any) -> None: ...
def async_dispatcher_send(hass: HomeAssistant, signal: SignalType[*_Ts,] | str, *args: *_Ts) -> None: ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_time_interval(hass: HomeAssistant, action: Callable[[datetime], Coroutine[Any, Any, None] | None], interval: timedelta, *, name: str | None = ..., cancel_on_shutdown: bool | None = ...) -> CALLBACK_TYPE: ...

# === Third-party dependency: ibeacon_ble ===
class iBeaconAdvertisement: ...
class iBeaconParser:
    ...
APPLE_MFR_ID: int
IBEACON_FIRST_BYTE: int
IBEACON_SECOND_BYTE: int