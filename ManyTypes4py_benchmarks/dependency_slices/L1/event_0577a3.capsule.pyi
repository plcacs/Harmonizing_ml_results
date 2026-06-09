# === Internal dependency: homeassistant.components.onvif.const ===
LOGGER = logging.getLogger(...)
DOMAIN = 'onvif'

# === Internal dependency: homeassistant.components.onvif.models ===
class Event: ...
class PullPointManagerState(Enum): ...
class WebHookManagerState(Enum): ...

# === Internal dependency: homeassistant.components.onvif.parsers ===
PARSERS = Registry(...)

# === Internal dependency: homeassistant.components.webhook ===
def async_register(hass, domain, name, webhook_id, handler, *, local_only=..., allowed_methods=...): ...
def async_unregister(hass, webhook_id): ...
def async_generate_path(webhook_id): ...

# === Internal dependency: homeassistant.core ===
def callback(func): ...
class HassJob:
    def __init__(self, target, name=..., *, cancel_on_shutdown=..., job_type=...): ...

# === Internal dependency: homeassistant.helpers.device_registry ===
def format_mac(mac): ...

# === Internal dependency: homeassistant.helpers.event ===
def async_call_later(hass, delay, action): ...

# === Internal dependency: homeassistant.helpers.network ===
class NoURLAvailableError(HomeAssistantError): ...
def get_url(hass, *, require_current_request=..., require_ssl=..., require_standard_port=..., require_cloud=..., allow_internal=..., allow_external=..., allow_cloud=..., allow_ip=..., prefer_external=..., prefer_cloud=...): ...

# === Internal dependency: homeassistant.util.decorator ===
class Registry(dict[_KT, _VT]): ...

# === Third-party dependency: httpx ===
# Used symbols: RemoteProtocolError, RequestError, TransportError

# === Unresolved dependency: onvif.client ===
# Used unresolved symbols: NotificationManager, PullPointManager, retry_connection_error

# === Unresolved dependency: onvif.exceptions ===
# Used unresolved symbols: ONVIFError

# === Unresolved dependency: onvif.util ===
# Used unresolved symbols: stringify_onvif_error

# === Unresolved dependency: zeep.exceptions ===
# Used unresolved symbols: Fault, ValidationError, XMLParseError