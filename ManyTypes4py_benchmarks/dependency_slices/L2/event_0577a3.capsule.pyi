from typing import Any

# === Internal dependency: homeassistant.components.onvif.const ===
LOGGER: getLogger
DOMAIN: str

# === Internal dependency: homeassistant.components.onvif.models ===
class Event: ...
class PullPointManagerState(Enum): ...
class WebHookManagerState(Enum): ...

# === Internal dependency: homeassistant.components.onvif.parsers ===
PARSERS: Registry[str, Callable[[str, Any], Coroutine[Any, Any, Event | None]]]

# === Internal dependency: homeassistant.components.webhook ===
def async_register(hass: HomeAssistant, domain: str, name: str, webhook_id: str, handler: Callable[[HomeAssistant, str, Request], Awaitable[Response | None]], *, local_only: bool | None = ..., allowed_methods: Iterable[str] | None = ...) -> None: ...
def async_unregister(hass: HomeAssistant, webhook_id: str) -> None: ...
def async_generate_path(webhook_id: str) -> str: ...

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...
class HassJob:
    def __init__(self, target: Callable[_P, _R_co], name: str | None = ..., *, cancel_on_shutdown: bool | None = ..., job_type: HassJobType | None = ...) -> None: ...

# === Internal dependency: homeassistant.helpers.device_registry ===
def format_mac(mac: str) -> str: ...

# === Internal dependency: homeassistant.helpers.event ===
def async_call_later(hass: HomeAssistant, delay: float | timedelta, action: HassJob[[datetime], Coroutine[Any, Any, None] | None] | Callable[[datetime], Coroutine[Any, Any, None] | None]) -> CALLBACK_TYPE: ...

# === Internal dependency: homeassistant.helpers.network ===
class NoURLAvailableError(HomeAssistantError): ...
def get_url(hass: HomeAssistant, *, require_current_request: bool = ..., require_ssl: bool = ..., require_standard_port: bool = ..., require_cloud: bool = ..., allow_internal: bool = ..., allow_external: bool = ..., allow_cloud: bool = ..., allow_ip: bool | None = ..., prefer_external: bool | None = ..., prefer_cloud: bool = ...) -> str: ...

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