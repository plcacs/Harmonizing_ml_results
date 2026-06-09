from typing import Any

# === Unresolved dependency: av ===
# Used unresolved symbols: logging

# === Internal dependency: homeassistant.components.stream.const ===
DOMAIN: str
ATTR_ENDPOINTS: str
ATTR_SETTINGS: str
ATTR_STREAMS: str
HLS_PROVIDER: str
RECORDER_PROVIDER: str
OUTPUT_IDLE_TIMEOUT: int
MAX_SEGMENTS: int
SEGMENT_DURATION_ADJUSTER: float
STREAM_RESTART_INCREMENT: int
STREAM_RESTART_RESET_TIME: int
CONF_LL_HLS: str
CONF_PART_DURATION: str
CONF_SEGMENT_DURATION: str
CONF_RTSP_TRANSPORT: str
RTSP_TRANSPORTS: Any
CONF_USE_WALLCLOCK_AS_TIMESTAMPS: str
CONF_EXTRA_PART_WAIT_TIME: str

# === Internal dependency: homeassistant.components.stream.core ===
PROVIDERS: Registry[str, type[StreamOutput]]
class StreamSettings:
    ...
class IdleTimer: ...
class StreamOutput: ...
class KeyFrameConverter:
    def __init__(self, hass: HomeAssistant, stream_settings: StreamSettings, dynamic_stream_settings: DynamicStreamSettings) -> None: ...
STREAM_SETTINGS_NON_LL_HLS: StreamSettings

# === Internal dependency: homeassistant.components.stream.diagnostics ===
class Diagnostics:
    def __init__(self) -> None: ...

# === Internal dependency: homeassistant.components.stream.hls ===
def async_setup_hls(hass: HomeAssistant) -> str: ...
class HlsStreamOutput(StreamOutput): ...

# === Internal dependency: homeassistant.components.stream.recorder ===
def async_setup_recorder(hass: HomeAssistant) -> None: ...
class RecorderOutput(StreamOutput): ...

# === Internal dependency: homeassistant.components.stream.worker ===
class StreamWorkerError(Exception): ...
class StreamState:
    def __init__(self, hass: HomeAssistant, outputs_callback: Callable[[], Mapping[str, StreamOutput]], diagnostics: Diagnostics) -> None: ...
    def discontinuity(self) -> None: ...
def stream_worker(source: str, pyav_options: dict[str, str], stream_settings: StreamSettings, stream_state: StreamState, keyframe_converter: KeyFrameConverter, quit_event: Event) -> None: ...

# === Internal dependency: homeassistant.const ===
EVENT_HOMEASSISTANT_STOP: EventType[NoEventData]
EVENT_LOGGING_CHANGED: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.exceptions ===
class HomeAssistantError(Exception): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def boolean(value: Any) -> bool: ...
positive_float: All

# === Internal dependency: homeassistant.setup ===
class SetupPhases(StrEnum): ...
def async_pause_setup(hass: core.HomeAssistant, phase: SetupPhases) -> Generator[None]: ...

# === Internal dependency: homeassistant.util.async_ ===
def create_eager_task(coro: Coroutine[Any, Any, _T], *, name: str | None = ..., loop: AbstractEventLoop | None = ...) -> Task[_T]: ...

# === Third-party dependency: voluptuous ===
# Used symbols: ALLOW_EXTRA, All, In, Invalid, Optional, Range, Schema

# === Third-party dependency: yarl ===
# Used symbols: URL