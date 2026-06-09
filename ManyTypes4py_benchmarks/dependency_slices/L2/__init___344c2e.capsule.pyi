from typing import Any

# === Third-party dependency: aiohttp ===
# Used symbols: BasicAuth

# === Third-party dependency: amcrest ===
class ApiWrapper(System, Network, MotionDetection, Snapshot, UserManagement, Event, Audio, Record, Video, Log, Ptz, Special, Storage, Nas, Media, PrivacyMode):
    ...
# re-export: from .exceptions import AmcrestError
# re-export: from .exceptions import LoginError

# === Internal dependency: homeassistant.auth.permissions.const ===
POLICY_CONTROL: str

# === Internal dependency: homeassistant.components.amcrest.binary_sensor ===
BINARY_SENSORS: tuple[AmcrestSensorEntityDescription, ...]
def check_binary_sensors(value: list[str]) -> list[str]: ...
BINARY_SENSOR_KEYS: Any

# === Internal dependency: homeassistant.components.amcrest.camera ===
STREAM_SOURCE_LIST: Any
CAMERA_SERVICES: Any

# === Internal dependency: homeassistant.components.amcrest.const ===
DOMAIN: str
DATA_AMCREST = DOMAIN
CAMERAS: str
DEVICES: str
COMM_RETRIES: int
COMM_TIMEOUT: float
SERVICE_EVENT: str
SERVICE_UPDATE: str
RESOLUTION_LIST: Any

# === Internal dependency: homeassistant.components.amcrest.helpers ===
def service_signal(service: str, *args: str) -> str: ...

# === Internal dependency: homeassistant.components.amcrest.sensor ===
SENSOR_KEYS: list[str]

# === Internal dependency: homeassistant.components.amcrest.switch ===
SWITCH_KEYS: list[str]

# === Internal dependency: homeassistant.const ===
class Platform(StrEnum): ...
ENTITY_MATCH_NONE: Final
ENTITY_MATCH_ALL: Final
CONF_AUTHENTICATION: Final
CONF_BINARY_SENSORS: Final
CONF_HOST: Final
CONF_NAME: Final
CONF_PASSWORD: Final
CONF_PORT: Final
CONF_SCAN_INTERVAL: Final
CONF_SENSORS: Final
CONF_SWITCHES: Final
CONF_USERNAME: Final
ATTR_ENTITY_ID: Final
HTTP_BASIC_AUTHENTICATION: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class Unauthorized(HomeAssistantError): ...
class UnknownUser(Unauthorized): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def boolean(value: Any) -> bool: ...
def ensure_list(value: None) -> list[Any]: ...
def ensure_list(value: list[_T]) -> list[_T]: ...
def ensure_list(value: list[_T] | _T) -> list[_T]: ...
def ensure_list(value: _T | None) -> list[_T] | list[Any]: ...
def string(value: Any) -> str: ...
port: All
time_period: Any

# === Internal dependency: homeassistant.helpers.discovery ===
async def async_load_platform(hass: core.HomeAssistant, component: Platform | str, platform: str, discovered: DiscoveryInfoType | None, hass_config: ConfigType) -> None: ...

# === Internal dependency: homeassistant.helpers.dispatcher ===
def dispatcher_send(hass: HomeAssistant, signal: SignalType[*_Ts,], *args: *_Ts) -> None: ...
def dispatcher_send(hass: HomeAssistant, signal: str, *args: Any) -> None: ...
def async_dispatcher_send(hass: HomeAssistant, signal: SignalType[*_Ts,], *args: *_Ts) -> None: ...
def async_dispatcher_send(hass: HomeAssistant, signal: str, *args: Any) -> None: ...
def async_dispatcher_send(hass: HomeAssistant, signal: SignalType[*_Ts,] | str, *args: *_Ts) -> None: ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_time_interval(hass: HomeAssistant, action: Callable[[datetime], Coroutine[Any, Any, None] | None], interval: timedelta, *, name: str | None = ..., cancel_on_shutdown: bool | None = ...) -> CALLBACK_TYPE: ...

# === Internal dependency: homeassistant.helpers.service ===
async def async_extract_entity_ids(hass: HomeAssistant, service_call: ServiceCall, expand_group: bool = ...) -> set[str]: ...

# === Third-party dependency: httpx ===
# Used symbols: Response

# === Third-party dependency: voluptuous ===
# Used symbols: ALLOW_EXTRA, All, Any, In, Optional, Required, Schema, Unique