from typing import Any

# === Internal dependency: homeassistant.components.binary_sensor ===
DOMAIN: str

# === Internal dependency: homeassistant.components.camera ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.homekit.const ===
DOMAIN: str
VIDEO_CODEC_COPY: str
VIDEO_CODEC_LIBX264: str
AUDIO_CODEC_OPUS: str
VIDEO_CODEC_H264_OMX: str
VIDEO_CODEC_H264_V4L2M2M: str
AUDIO_CODEC_COPY: str
CONF_AUDIO_CODEC: str
CONF_AUDIO_MAP: str
CONF_AUDIO_PACKET_SIZE: str
CONF_FEATURE: str
CONF_FEATURE_LIST: str
CONF_LINKED_BATTERY_SENSOR: str
CONF_LINKED_BATTERY_CHARGING_SENSOR: str
CONF_LINKED_DOORBELL_SENSOR: str
CONF_LINKED_MOTION_SENSOR: str
CONF_LINKED_HUMIDITY_SENSOR: str
CONF_LINKED_OBSTRUCTION_SENSOR: str
CONF_LOW_BATTERY_THRESHOLD: str
CONF_MAX_FPS: str
CONF_MAX_HEIGHT: str
CONF_MAX_WIDTH: str
CONF_STREAM_ADDRESS: str
CONF_STREAM_SOURCE: str
CONF_SUPPORT_AUDIO: str
CONF_THRESHOLD_CO: str
CONF_THRESHOLD_CO2: str
CONF_VIDEO_CODEC: str
CONF_VIDEO_MAP: str
CONF_VIDEO_PACKET_SIZE: str
CONF_STREAM_COUNT: str
DEFAULT_SUPPORT_AUDIO: bool
DEFAULT_AUDIO_CODEC = AUDIO_CODEC_OPUS
DEFAULT_AUDIO_MAP: str
DEFAULT_AUDIO_PACKET_SIZE: int
DEFAULT_LOW_BATTERY_THRESHOLD: int
DEFAULT_MAX_FPS: int
DEFAULT_MAX_HEIGHT: int
DEFAULT_MAX_WIDTH: int
DEFAULT_VIDEO_CODEC = VIDEO_CODEC_LIBX264
DEFAULT_VIDEO_MAP: str
DEFAULT_VIDEO_PACKET_SIZE: int
DEFAULT_STREAM_COUNT: int
FEATURE_ON_OFF: str
FEATURE_PLAY_PAUSE: str
FEATURE_PLAY_STOP: str
FEATURE_TOGGLE_MUTE: str
TYPE_FAUCET: str
TYPE_OUTLET: str
TYPE_SHOWER: str
TYPE_SPRINKLER: str
TYPE_SWITCH: str
TYPE_VALVE: str
MAX_NAME_LENGTH: int

# === Internal dependency: homeassistant.components.homekit.models ===
class HomeKitEntryData: ...

# === Internal dependency: homeassistant.components.lock ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.components.media_player ===
class MediaPlayerDeviceClass(StrEnum): ...
# re-export: from .const import DOMAIN
# re-export: from .const import MediaPlayerEntityFeature

# === Internal dependency: homeassistant.components.persistent_notification ===
def async_create(hass: HomeAssistant, message: str, title: str | None = ..., notification_id: str | None = ...) -> None: ...
def async_dismiss(hass: HomeAssistant, notification_id: str) -> None: ...

# === Internal dependency: homeassistant.components.remote ===
class RemoteEntityFeature(IntFlag): ...
DOMAIN: str

# === Internal dependency: homeassistant.components.sensor ===
# re-export: from .const import DOMAIN

# === Internal dependency: homeassistant.const ===
CONF_NAME: Final
CONF_PORT: Final
CONF_TYPE: Final
ATTR_CODE: Final
ATTR_SUPPORTED_FEATURES: Final
ATTR_DEVICE_CLASS: Final
class UnitOfTemperature(StrEnum): ...

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id: str) -> tuple[str, str]: ...
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def boolean(value: Any) -> bool: ...
def ensure_list(value: None) -> list[Any]: ...
def ensure_list(value: list[_T]) -> list[_T]: ...
def ensure_list(value: list[_T] | _T) -> list[_T]: ...
def ensure_list(value: _T | None) -> list[_T] | list[Any]: ...
def entity_id(value: Any) -> str: ...
def entity_domain(domain: str | list[str]) -> Callable[[Any], str]: ...
def string(value: Any) -> str: ...
positive_int: All

# === Internal dependency: homeassistant.helpers.storage ===
STORAGE_DIR: str

# === Internal dependency: homeassistant.util.unit_conversion ===
class TemperatureConverter(BaseUnitConverter): ...

# === Third-party dependency: pyqrcode ===
def create(content, error = ..., version = ..., mode = ..., encoding = ...) -> Any: ...

# === Third-party dependency: voluptuous ===
# Used symbols: All, Any, Coerce, In, Invalid, Optional, Range, Required, Schema