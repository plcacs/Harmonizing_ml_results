# === Internal dependency: homeassistant.components.bluetooth ===
from habluetooth import BluetoothScanningMode
from home_assistant_bluetooth import BluetoothServiceInfoBleak
from .api import async_last_service_info
from .api import async_register_callback
from .api import async_track_unavailable
from .models import BluetoothChange

# === Internal dependency: homeassistant.components.bluetooth.match ===
class BluetoothCallbackMatcher(BluetoothMatcherOptional, BluetoothCallbackMatcherOptional): ...

# === Internal dependency: homeassistant.components.ibeacon.const ===
DOMAIN = 'ibeacon'
SIGNAL_IBEACON_DEVICE_NEW = 'ibeacon_tracker_new_device'
SIGNAL_IBEACON_DEVICE_UNAVAILABLE = 'ibeacon_tracker_unavailable_device'
SIGNAL_IBEACON_DEVICE_SEEN = 'ibeacon_seen_device'
UNAVAILABLE_TIMEOUT = 180
UPDATE_INTERVAL = timedelta(...)
MAX_IDS = 10
MAX_IDS_PER_UUID = 50
MIN_SEEN_TRANSIENT_NEW = round(FALLBACK_MAXIMUM_STALE_ADVERTISEMENT_SECONDS / UPDATE_INTERVAL.total_seconds()) + 1
CONF_IGNORE_ADDRESSES = 'ignore_addresses'
CONF_IGNORE_UUIDS = 'ignore_uuids'
CONF_ALLOW_NAMELESS_UUIDS = 'allow_nameless_uuids'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.helpers.dispatcher ===
def async_dispatcher_send(hass, signal, *args): ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_time_interval(hass, action, interval, *, name=..., cancel_on_shutdown=...): ...

# === Third-party dependency: ibeacon_ble ===
class iBeaconAdvertisement: ...
class iBeaconParser:
    ...
APPLE_MFR_ID: int
IBEACON_FIRST_BYTE: int
IBEACON_SECOND_BYTE: int