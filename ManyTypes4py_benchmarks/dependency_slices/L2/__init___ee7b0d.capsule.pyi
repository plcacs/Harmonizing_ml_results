from typing import Any

# === Third-party dependency: aiohttp.web ===
# Used symbols: Response

# === Internal dependency: homeassistant.components.camera ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.media_source ===
# re-export: from .const import URI_SCHEME

# === Internal dependency: homeassistant.components.motioneye.const ===
DOMAIN: Final
ATTR_EVENT_TYPE: Final
ATTR_WEBHOOK_ID: Final
CONF_CLIENT: Final
CONF_COORDINATOR: Final
CONF_ADMIN_PASSWORD: Final
CONF_ADMIN_USERNAME: Final
CONF_SURVEILLANCE_USERNAME: Final
CONF_SURVEILLANCE_PASSWORD: Final
CONF_WEBHOOK_SET: Final
CONF_WEBHOOK_SET_OVERWRITE: Final
DEFAULT_WEBHOOK_SET: Final
DEFAULT_WEBHOOK_SET_OVERWRITE: Final
DEFAULT_SCAN_INTERVAL: Final
EVENT_MOTION_DETECTED: Final
EVENT_FILE_STORED: Final
EVENT_MOTION_DETECTED_KEYS: Final
EVENT_FILE_STORED_KEYS: Final
EVENT_MEDIA_CONTENT_ID: Final
MOTIONEYE_MANUFACTURER: Final
SIGNAL_CAMERA_ADD: Final
WEB_HOOK_SENTINEL_KEY: Final
WEB_HOOK_SENTINEL_VALUE: Final

# === Internal dependency: homeassistant.components.sensor ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.switch ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.webhook ===
def async_register(hass: HomeAssistant, domain: str, name: str, webhook_id: str, handler: Callable[[HomeAssistant, str, Request], Awaitable[Response | None]], *, local_only: bool | None = ..., allowed_methods: Iterable[str] | None = ...) -> None: ...
def async_unregister(hass: HomeAssistant, webhook_id: str) -> None: ...
def async_generate_id() -> str: ...
def async_generate_path(webhook_id: str) -> str: ...

# === Internal dependency: homeassistant.const ===
CONF_URL: Final
CONF_WEBHOOK_ID: Final
ATTR_NAME: Final
ATTR_DEVICE_ID: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class ConfigEntryNotReady(IntegrationError): ...
class ConfigEntryAuthFailed(IntegrationError): ...

# === Internal dependency: homeassistant.helpers.aiohttp_client ===
def async_get_clientsession(hass: HomeAssistant, verify_ssl: bool = ..., family: int = ...) -> aiohttp.ClientSession: ...

# === Internal dependency: homeassistant.helpers.device_registry ===
class DeviceInfo(TypedDict): ...
def async_get(hass: HomeAssistant) -> DeviceRegistry: ...
def async_entries_for_config_entry(registry: DeviceRegistry, config_entry_id: str) -> list[DeviceEntry]: ...

# === Internal dependency: homeassistant.helpers.dispatcher ===
def async_dispatcher_connect(hass: HomeAssistant, signal: SignalType[*_Ts,], target: Callable[[*_Ts], Any]) -> Callable[[], None]: ...
def async_dispatcher_connect(hass: HomeAssistant, signal: str, target: Callable[..., Any]) -> Callable[[], None]: ...
def async_dispatcher_connect(hass: HomeAssistant, signal: SignalType[*_Ts,] | str, target: Callable[[*_Ts], Any] | Callable[..., Any]) -> Callable[[], None]: ...
def async_dispatcher_send(hass: HomeAssistant, signal: SignalType[*_Ts,], *args: *_Ts) -> None: ...
def async_dispatcher_send(hass: HomeAssistant, signal: str, *args: Any) -> None: ...
def async_dispatcher_send(hass: HomeAssistant, signal: SignalType[*_Ts,] | str, *args: *_Ts) -> None: ...

# === Internal dependency: homeassistant.helpers.network ===
class NoURLAvailableError(HomeAssistantError): ...
def get_url(hass: HomeAssistant, *, require_current_request: bool = ..., require_ssl: bool = ..., require_standard_port: bool = ..., require_cloud: bool = ..., allow_internal: bool = ..., allow_external: bool = ..., allow_cloud: bool = ..., allow_ip: bool | None = ..., prefer_external: bool | None = ..., prefer_cloud: bool = ...) -> str: ...

# === Internal dependency: homeassistant.helpers.update_coordinator ===
class UpdateFailed(Exception): ...
class DataUpdateCoordinator(BaseDataUpdateCoordinatorProtocol, Generic[_DataT]):
    def __init__(self, hass: HomeAssistant, logger: logging.Logger, *, name: str, update_interval: timedelta | None = ..., update_method: Callable[[], Awaitable[_DataT]] | None = ..., request_refresh_debouncer: Debouncer[Coroutine[Any, Any, None]] | None = ..., always_update: bool = ...) -> None: ...
    def async_add_listener(self, update_callback: CALLBACK_TYPE, context: Any = ...) -> Callable[[], None]: ...
    async def async_refresh(self) -> None: ...
class CoordinatorEntity(BaseCoordinatorEntity[_DataUpdateCoordinatorT]):
    def available(self) -> bool: ...

# === Third-party dependency: motioneye_client.client ===
class MotionEyeClientError(Exception): ...
class MotionEyeClientInvalidAuthError(MotionEyeClientError): ...
class MotionEyeClientPathError(MotionEyeClientError): ...
class MotionEyeClient: ...

# === Third-party dependency: motioneye_client.const ===
KEY_CAMERAS: str
KEY_ID: str
KEY_NAME: str
KEY_ROOT_DIRECTORY: str
KEY_WEB_HOOK_NOTIFICATIONS_ENABLED: str
KEY_WEB_HOOK_NOTIFICATIONS_HTTP_METHOD: str
KEY_WEB_HOOK_NOTIFICATIONS_URL: str
KEY_WEB_HOOK_STORAGE_ENABLED: str
KEY_WEB_HOOK_STORAGE_HTTP_METHOD: str
KEY_WEB_HOOK_STORAGE_URL: str
KEY_HTTP_METHOD_POST_JSON: str
KEY_WEB_HOOK_CS_FILE_PATH: str
KEY_WEB_HOOK_CS_FILE_TYPE: str
KEY_WEB_HOOK_CONVERSION_SPECIFIERS: Any