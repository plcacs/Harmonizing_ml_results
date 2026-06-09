from typing import Any

# === Internal dependency: homeassistant.components.binary_sensor ===
DOMAIN = 'binary_sensor'

# === Internal dependency: homeassistant.components.camera ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.homekit.const ===
DOMAIN = 'homekit'
VIDEO_CODEC_COPY = 'copy'
VIDEO_CODEC_LIBX264 = 'libx264'
AUDIO_CODEC_OPUS = 'libopus'
VIDEO_CODEC_H264_OMX = 'h264_omx'
VIDEO_CODEC_H264_V4L2M2M = 'h264_v4l2m2m'
AUDIO_CODEC_COPY = 'copy'
CONF_AUDIO_CODEC = 'audio_codec'
CONF_AUDIO_MAP = 'audio_map'
CONF_AUDIO_PACKET_SIZE = 'audio_packet_size'
CONF_FEATURE = 'feature'
CONF_FEATURE_LIST = 'feature_list'
CONF_LINKED_BATTERY_SENSOR = 'linked_battery_sensor'
CONF_LINKED_BATTERY_CHARGING_SENSOR = 'linked_battery_charging_sensor'
CONF_LINKED_DOORBELL_SENSOR = 'linked_doorbell_sensor'
CONF_LINKED_MOTION_SENSOR = 'linked_motion_sensor'
CONF_LINKED_HUMIDITY_SENSOR = 'linked_humidity_sensor'
CONF_LINKED_OBSTRUCTION_SENSOR = 'linked_obstruction_sensor'
CONF_LOW_BATTERY_THRESHOLD = 'low_battery_threshold'
CONF_MAX_FPS = 'max_fps'
CONF_MAX_HEIGHT = 'max_height'
CONF_MAX_WIDTH = 'max_width'
CONF_STREAM_ADDRESS = 'stream_address'
CONF_STREAM_SOURCE = 'stream_source'
CONF_SUPPORT_AUDIO = 'support_audio'
CONF_THRESHOLD_CO = 'co_threshold'
CONF_THRESHOLD_CO2 = 'co2_threshold'
CONF_VIDEO_CODEC = 'video_codec'
CONF_VIDEO_MAP = 'video_map'
CONF_VIDEO_PACKET_SIZE = 'video_packet_size'
CONF_STREAM_COUNT = 'stream_count'
DEFAULT_SUPPORT_AUDIO = False
DEFAULT_AUDIO_CODEC = AUDIO_CODEC_OPUS
DEFAULT_AUDIO_MAP = '0:a:0'
DEFAULT_AUDIO_PACKET_SIZE = 188
DEFAULT_LOW_BATTERY_THRESHOLD = 20
DEFAULT_MAX_FPS = 30
DEFAULT_MAX_HEIGHT = 1080
DEFAULT_MAX_WIDTH = 1920
DEFAULT_VIDEO_CODEC = VIDEO_CODEC_LIBX264
DEFAULT_VIDEO_MAP = '0:v:0'
DEFAULT_VIDEO_PACKET_SIZE = 1316
DEFAULT_STREAM_COUNT = 3
FEATURE_ON_OFF = 'on_off'
FEATURE_PLAY_PAUSE = 'play_pause'
FEATURE_PLAY_STOP = 'play_stop'
FEATURE_TOGGLE_MUTE = 'toggle_mute'
TYPE_FAUCET = 'faucet'
TYPE_OUTLET = 'outlet'
TYPE_SHOWER = 'shower'
TYPE_SPRINKLER = 'sprinkler'
TYPE_SWITCH = 'switch'
TYPE_VALVE = 'valve'
MAX_NAME_LENGTH = 64

# === Internal dependency: homeassistant.components.homekit.models ===
class HomeKitEntryData: ...

# === Internal dependency: homeassistant.components.lock ===
from .const import DOMAIN

# === Internal dependency: homeassistant.components.media_player ===
class MediaPlayerDeviceClass(StrEnum): ...
from .const import DOMAIN
from .const import MediaPlayerEntityFeature

# === Internal dependency: homeassistant.components.persistent_notification ===
def async_create(hass, message, title=..., notification_id=...): ...
def async_dismiss(hass, notification_id): ...

# === Internal dependency: homeassistant.components.remote ===
class RemoteEntityFeature(IntFlag): ...
DOMAIN = 'remote'

# === Internal dependency: homeassistant.components.sensor ===
from .const import DOMAIN

# === Internal dependency: homeassistant.const ===
class UnitOfTemperature(StrEnum): ...
CONF_NAME = 'name'
CONF_PORT = 'port'
CONF_TYPE = 'type'
ATTR_CODE = 'code'
ATTR_SUPPORTED_FEATURES = 'supported_features'
ATTR_DEVICE_CLASS = 'device_class'

# === Internal dependency: homeassistant.core ===
def split_entity_id(entity_id): ...
def callback(func): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def boolean(value): ...
def ensure_list(value): ...
def entity_id(value): ...
def entity_domain(domain): ...
def string(value): ...
positive_int = vol.All(...)

# === Internal dependency: homeassistant.helpers.storage ===
STORAGE_DIR = '.storage'

# === Internal dependency: homeassistant.util.unit_conversion ===
class TemperatureConverter(BaseUnitConverter): ...

# === Third-party dependency: pyqrcode ===
def create(content, error = ..., version = ..., mode = ..., encoding = ...) -> Any: ...

# === Third-party dependency: voluptuous ===
# Used symbols: All, Any, Coerce, In, Invalid, Optional, Range, Required, Schema