from typing import Any

# === Third-party dependency: aiohttp ===
# Used symbols: BasicAuth

# === Third-party dependency: amcrest ===
class ApiWrapper(System, Network, MotionDetection, Snapshot, UserManagement, Event, Audio, Record, Video, Log, Ptz, Special, Storage, Nas, Media, PrivacyMode):
    ...
# re-export: from .exceptions import AmcrestError
# re-export: from .exceptions import LoginError

# === Internal dependency: homeassistant.auth.permissions.const ===
POLICY_CONTROL = 'control'

# === Internal dependency: homeassistant.components.amcrest.binary_sensor ===
class AmcrestSensorEntityDescription(BinarySensorEntityDescription):
    ...
def check_binary_sensors(value): ...
_AUDIO_DETECTED_KEY = 'audio_detected'
_AUDIO_DETECTED_POLLED_KEY = 'audio_detected_polled'
_AUDIO_DETECTED_NAME = 'Audio Detected'
_AUDIO_DETECTED_EVENT_CODES = {'AudioMutation', 'AudioIntensity'}
_CROSSLINE_DETECTED_KEY = 'crossline_detected'
_CROSSLINE_DETECTED_POLLED_KEY = 'crossline_detected_polled'
_CROSSLINE_DETECTED_NAME = 'CrossLine Detected'
_CROSSLINE_DETECTED_EVENT_CODE = 'CrossLineDetection'
_MOTION_DETECTED_KEY = 'motion_detected'
_MOTION_DETECTED_POLLED_KEY = 'motion_detected_polled'
_MOTION_DETECTED_NAME = 'Motion Detected'
_MOTION_DETECTED_EVENT_CODE = 'VideoMotion'
_ONLINE_KEY = 'online'
BINARY_SENSORS = (AmcrestSensorEntityDescription(...), AmcrestSensorEntityDescription(...), AmcrestSensorEntityDescription(...), AmcrestSensorEntityDescription(...), AmcrestSensorEntityDescription(...), AmcrestSensorEntityDescription(...), AmcrestSensorEntityDescription(...))
BINARY_SENSOR_KEYS = [description.key for description in BINARY_SENSORS]

# === Internal dependency: homeassistant.components.amcrest.camera ===
STREAM_SOURCE_LIST = ['snapshot', 'mjpeg', 'rtsp']
_SRV_EN_REC = 'enable_recording'
_SRV_DS_REC = 'disable_recording'
_SRV_EN_AUD = 'enable_audio'
_SRV_DS_AUD = 'disable_audio'
_SRV_EN_MOT_REC = 'enable_motion_recording'
_SRV_DS_MOT_REC = 'disable_motion_recording'
_SRV_GOTO = 'goto_preset'
_SRV_CBW = 'set_color_bw'
_SRV_TOUR_ON = 'start_tour'
_SRV_TOUR_OFF = 'stop_tour'
_SRV_PTZ_CTRL = 'ptz_control'
_ATTR_PTZ_TT = 'travel_time'
_ATTR_PTZ_MOV = 'movement'
_MOV = ['zoom_out', 'zoom_in', 'right', 'left', 'up', 'down', 'right_down', 'right_up', ...]
_DEFAULT_TT = 0.2
_ATTR_PRESET = 'preset'
_ATTR_COLOR_BW = 'color_bw'
_CBW_COLOR = 'color'
_CBW_AUTO = 'auto'
_CBW_BW = 'bw'
_CBW = [_CBW_COLOR, _CBW_AUTO, _CBW_BW]
_SRV_SCHEMA = vol.Schema(...)
_SRV_GOTO_SCHEMA = _SRV_SCHEMA.extend(...)
_SRV_CBW_SCHEMA = _SRV_SCHEMA.extend(...)
_SRV_PTZ_SCHEMA = _SRV_SCHEMA.extend(...)
CAMERA_SERVICES = {_SRV_EN_REC: (_SRV_SCHEMA, 'async_enable_recording', ()), _SRV_DS_REC: (_SRV_SCHEMA, 'async_disable_recording', ()), _SRV_EN_AUD: (_SRV_SCHEMA, 'async_enable_audio', ()), _SRV_DS_AUD: (_SRV_SCHEMA, 'async_disable_audio', ()), _SRV_EN_MOT_REC: (_SRV_SCHEMA, 'async_enable_motion_recording', ()), _SRV_DS_MOT_REC: (_SRV_SCHEMA, 'async_disable_motion_recording', ()), _SRV_GOTO: (_SRV_GOTO_SCHEMA, 'async_goto_preset', (_ATTR_PRESET,)), _SRV_CBW: (_SRV_CBW_SCHEMA, 'async_set_color_bw', (_ATTR_COLOR_BW,)), ...}

# === Internal dependency: homeassistant.components.amcrest.const ===
DOMAIN = 'amcrest'
DATA_AMCREST = DOMAIN
CAMERAS = 'cameras'
DEVICES = 'devices'
COMM_RETRIES = 1
COMM_TIMEOUT = 6.05
SERVICE_EVENT = 'event'
SERVICE_UPDATE = 'update'
RESOLUTION_LIST = {'high': 0, 'low': 1}

# === Internal dependency: homeassistant.components.amcrest.helpers ===
def service_signal(service, *args): ...

# === Internal dependency: homeassistant.components.amcrest.sensor ===
SENSOR_PTZ_PRESET = 'ptz_preset'
SENSOR_SDCARD = 'sdcard'
SENSOR_TYPES = (SensorEntityDescription(...), SensorEntityDescription(...))
SENSOR_KEYS = [desc.key for desc in SENSOR_TYPES]

# === Internal dependency: homeassistant.components.amcrest.switch ===
PRIVACY_MODE_KEY = 'privacy_mode'
SWITCH_TYPES = (SwitchEntityDescription(...),)
SWITCH_KEYS = [desc.key for desc in SWITCH_TYPES]

# === Internal dependency: homeassistant.const ===
class Platform(StrEnum): ...
ENTITY_MATCH_NONE = 'none'
ENTITY_MATCH_ALL = 'all'
CONF_AUTHENTICATION = 'authentication'
CONF_BINARY_SENSORS = 'binary_sensors'
CONF_HOST = 'host'
CONF_NAME = 'name'
CONF_PASSWORD = 'password'
CONF_PORT = 'port'
CONF_SCAN_INTERVAL = 'scan_interval'
CONF_SENSORS = 'sensors'
CONF_SWITCHES = 'switches'
CONF_USERNAME = 'username'
ATTR_ENTITY_ID = 'entity_id'
HTTP_BASIC_AUTHENTICATION = 'basic'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.exceptions ===
class Unauthorized(HomeAssistantError): ...
class UnknownUser(Unauthorized): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def has_at_least_one_key(*keys): ...
def boolean(value): ...
def ensure_list(value): ...
def time_period_str(value): ...
def time_period_seconds(value): ...
def string(value): ...
port = vol.All(...)
_TIME_PERIOD_DICT_KEYS = ('days', 'hours', 'minutes', 'seconds', 'milliseconds')
time_period_dict = vol.All(...)
time_period = vol.Any(...)

# === Internal dependency: homeassistant.helpers.discovery ===
async def async_load_platform(hass, component, platform, discovered, hass_config): ...

# === Internal dependency: homeassistant.helpers.dispatcher ===
def dispatcher_send(hass, signal, *args): ...
def async_dispatcher_send(hass, signal, *args): ...

# === Internal dependency: homeassistant.helpers.event ===
def async_track_time_interval(hass, action, interval, *, name=..., cancel_on_shutdown=...): ...

# === Internal dependency: homeassistant.helpers.service ===
async def async_extract_entity_ids(hass, service_call, expand_group=...): ...

# === Third-party dependency: httpx ===
# Used symbols: Response

# === Third-party dependency: voluptuous ===
# Used symbols: ALLOW_EXTRA, All, Any, In, Optional, Required, Schema, Unique