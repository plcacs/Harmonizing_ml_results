# === Unresolved dependency: av ===
# Used unresolved symbols: logging

# === Internal dependency: homeassistant.components.stream.const ===
DOMAIN = 'stream'
ATTR_ENDPOINTS = 'endpoints'
ATTR_SETTINGS = 'settings'
ATTR_STREAMS = 'streams'
HLS_PROVIDER = 'hls'
RECORDER_PROVIDER = 'recorder'
OUTPUT_IDLE_TIMEOUT = 300
MAX_SEGMENTS = 5
SEGMENT_DURATION_ADJUSTER = 0.1
STREAM_RESTART_INCREMENT = 10
STREAM_RESTART_RESET_TIME = 300
CONF_LL_HLS = 'll_hls'
CONF_PART_DURATION = 'part_duration'
CONF_SEGMENT_DURATION = 'segment_duration'
CONF_RTSP_TRANSPORT = 'rtsp_transport'
RTSP_TRANSPORTS = {'tcp': 'TCP', 'udp': 'UDP', 'udp_multicast': 'UDP Multicast', 'http': 'HTTP'}
CONF_USE_WALLCLOCK_AS_TIMESTAMPS = 'use_wallclock_as_timestamps'
CONF_EXTRA_PART_WAIT_TIME = 'extra_part_wait_time'

# === Internal dependency: homeassistant.components.stream.core ===
class StreamSettings:
    ...
class IdleTimer: ...
class StreamOutput: ...
class KeyFrameConverter:
    def __init__(self, hass, stream_settings, dynamic_stream_settings): ...
PROVIDERS = Registry(...)
STREAM_SETTINGS_NON_LL_HLS = StreamSettings(...)

# === Internal dependency: homeassistant.components.stream.diagnostics ===
class Diagnostics:
    def __init__(self): ...

# === Internal dependency: homeassistant.components.stream.hls ===
def async_setup_hls(hass): ...
class HlsStreamOutput(StreamOutput): ...

# === Internal dependency: homeassistant.components.stream.recorder ===
def async_setup_recorder(hass): ...
class RecorderOutput(StreamOutput): ...

# === Internal dependency: homeassistant.components.stream.worker ===
class StreamWorkerError(Exception): ...
class StreamState:
    def __init__(self, hass, outputs_callback, diagnostics): ...
    def discontinuity(self): ...
def stream_worker(source, pyav_options, stream_settings, stream_state, keyframe_converter, quit_event): ...

# === Internal dependency: homeassistant.const ===
EVENT_HOMEASSISTANT_STOP = EventType(...)
EVENT_LOGGING_CHANGED = 'logging_changed'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def boolean(value): ...
positive_float = vol.All(...)

# === Internal dependency: homeassistant.setup ===
class SetupPhases(StrEnum): ...
def async_pause_setup(hass, phase): ...

# === Internal dependency: homeassistant.util.async_ ===
def create_eager_task(coro, *, name=..., loop=...): ...

# === Internal dependency: homeassistant.util.decorator ===
class Registry(dict[_KT, _VT]): ...

# === Internal dependency: homeassistant.util.event_type ===
class EventType(Generic[_DataT]): ...

# === Third-party dependency: voluptuous ===
# Used symbols: ALLOW_EXTRA, All, In, Invalid, Optional, Range, Schema

# === Third-party dependency: yarl ===
# Used symbols: URL