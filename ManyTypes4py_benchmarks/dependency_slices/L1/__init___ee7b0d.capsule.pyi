from typing import Any

# === Third-party dependency: aiohttp.web ===
# Used symbols: Response

# === Internal dependency: homeassistant.components.camera ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.media_source ===
from .const import URI_SCHEME

# === Internal dependency: homeassistant.components.motioneye.const ===
DOMAIN = 'motioneye'
ATTR_EVENT_TYPE = 'event_type'
ATTR_WEBHOOK_ID = 'webhook_id'
CONF_CLIENT = 'client'
CONF_COORDINATOR = 'coordinator'
CONF_ADMIN_PASSWORD = 'admin_password'
CONF_ADMIN_USERNAME = 'admin_username'
CONF_SURVEILLANCE_USERNAME = 'surveillance_username'
CONF_SURVEILLANCE_PASSWORD = 'surveillance_password'
CONF_WEBHOOK_SET = 'webhook_set'
CONF_WEBHOOK_SET_OVERWRITE = 'webhook_set_overwrite'
DEFAULT_WEBHOOK_SET = True
DEFAULT_WEBHOOK_SET_OVERWRITE = False
DEFAULT_SCAN_INTERVAL = timedelta(...)
EVENT_MOTION_DETECTED = 'motion_detected'
EVENT_FILE_STORED = 'file_stored'
EVENT_MOTION_DETECTED_KEYS = [KEY_WEB_HOOK_CS_EVENT, KEY_WEB_HOOK_CS_FRAME_NUMBER, KEY_WEB_HOOK_CS_CAMERA_ID, KEY_WEB_HOOK_CS_CHANGED_PIXELS, KEY_WEB_HOOK_CS_NOISE_LEVEL, KEY_WEB_HOOK_CS_WIDTH, KEY_WEB_HOOK_CS_HEIGHT, KEY_WEB_HOOK_CS_MOTION_WIDTH, ...]
EVENT_FILE_STORED_KEYS = [KEY_WEB_HOOK_CS_EVENT, KEY_WEB_HOOK_CS_FRAME_NUMBER, KEY_WEB_HOOK_CS_CAMERA_ID, KEY_WEB_HOOK_CS_NOISE_LEVEL, KEY_WEB_HOOK_CS_WIDTH, KEY_WEB_HOOK_CS_HEIGHT, KEY_WEB_HOOK_CS_FILE_PATH, KEY_WEB_HOOK_CS_FILE_TYPE, ...]
EVENT_MEDIA_CONTENT_ID = 'media_content_id'
MOTIONEYE_MANUFACTURER = 'motionEye'
SIGNAL_CAMERA_ADD = f'{DOMAIN}_camera_add_signal.{{}}'
WEB_HOOK_SENTINEL_KEY = 'src'
WEB_HOOK_SENTINEL_VALUE = 'hass-motioneye'

# === Internal dependency: homeassistant.components.sensor ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.switch ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.webhook ===
def async_register(hass, domain, name, webhook_id, handler, *, local_only=..., allowed_methods=...): ...
def async_unregister(hass, webhook_id): ...
def async_generate_id(): ...
def async_generate_path(webhook_id): ...

# === Internal dependency: homeassistant.const ===
CONF_URL = 'url'
CONF_WEBHOOK_ID = 'webhook_id'
ATTR_NAME = 'name'
ATTR_DEVICE_ID = 'device_id'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.exceptions ===
class ConfigEntryNotReady(IntegrationError): ...
class ConfigEntryAuthFailed(IntegrationError): ...

# === Internal dependency: homeassistant.helpers.aiohttp_client ===
def async_get_clientsession(hass, verify_ssl=..., family=...): ...

# === Internal dependency: homeassistant.helpers.device_registry ===
class DeviceInfo(TypedDict): ...
def async_get(hass): ...
def async_entries_for_config_entry(registry, config_entry_id): ...

# === Internal dependency: homeassistant.helpers.dispatcher ===
def async_dispatcher_connect(hass, signal, target): ...
def async_dispatcher_send(hass, signal, *args): ...

# === Internal dependency: homeassistant.helpers.network ===
class NoURLAvailableError(HomeAssistantError): ...
def get_url(hass, *, require_current_request=..., require_ssl=..., require_standard_port=..., require_cloud=..., allow_internal=..., allow_external=..., allow_cloud=..., allow_ip=..., prefer_external=..., prefer_cloud=...): ...

# === Internal dependency: homeassistant.helpers.update_coordinator ===
class UpdateFailed(Exception): ...
class DataUpdateCoordinator(BaseDataUpdateCoordinatorProtocol, Generic[_DataT]):
    def __init__(self, hass, logger, *, name, update_interval=..., update_method=..., request_refresh_debouncer=..., always_update=...): ...
    def async_add_listener(self, update_callback, context=...): ...
    async def async_refresh(self): ...
class CoordinatorEntity(BaseCoordinatorEntity[_DataUpdateCoordinatorT]):
    def available(self): ...

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