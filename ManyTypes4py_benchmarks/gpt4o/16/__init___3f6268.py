from __future__ import annotations
import asyncio
from collections.abc import Callable, Mapping
import copy
import logging
import secrets
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Optional, Dict, Union, List
import voluptuous as vol
from yarl import URL
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, EVENT_LOGGING_CHANGED
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType
from homeassistant.setup import SetupPhases, async_pause_setup
from homeassistant.util.async_ import create_eager_task
from .const import (
    ATTR_ENDPOINTS, ATTR_PREFER_TCP, ATTR_SETTINGS, ATTR_STREAMS,
    CONF_EXTRA_PART_WAIT_TIME, CONF_LL_HLS, CONF_PART_DURATION,
    CONF_RTSP_TRANSPORT, CONF_SEGMENT_DURATION, CONF_USE_WALLCLOCK_AS_TIMESTAMPS,
    DOMAIN, FORMAT_CONTENT_TYPE, HLS_PROVIDER, MAX_SEGMENTS, OUTPUT_FORMATS,
    OUTPUT_IDLE_TIMEOUT, RECORDER_PROVIDER, RTSP_TRANSPORTS,
    SEGMENT_DURATION_ADJUSTER, SOURCE_TIMEOUT, STREAM_RESTART_INCREMENT,
    STREAM_RESTART_RESET_TIME, StreamClientError
)
from .core import (
    PROVIDERS, STREAM_SETTINGS_NON_LL_HLS, IdleTimer, KeyFrameConverter,
    Orientation, StreamOutput, StreamSettings
)
from .diagnostics import Diagnostics
from .exceptions import StreamOpenClientError, StreamWorkerError
from .hls import HlsStreamOutput, async_setup_hls

if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings

__all__ = [
    'ATTR_SETTINGS', 'CONF_EXTRA_PART_WAIT_TIME', 'CONF_RTSP_TRANSPORT',
    'CONF_USE_WALLCLOCK_AS_TIMESTAMPS', 'DOMAIN', 'FORMAT_CONTENT_TYPE',
    'HLS_PROVIDER', 'OUTPUT_FORMATS', 'RTSP_TRANSPORTS', 'SOURCE_TIMEOUT',
    'Orientation', 'Stream', 'StreamClientError', 'StreamOpenClientError',
    'create_stream'
]

_LOGGER = logging.getLogger(__name__)

async def async_check_stream_client_error(
    hass: HomeAssistant, source: str, pyav_options: Optional[Dict[str, Any]] = None
) -> None:
    """Check if a stream can be successfully opened.

    Raise StreamOpenClientError if an http client error is encountered.
    """
    await hass.loop.run_in_executor(None, _check_stream_client_error, hass, source, pyav_options)

def _check_stream_client_error(
    hass: HomeAssistant, source: str, options: Optional[Dict[str, Any]] = None
) -> None:
    """Check if a stream can be successfully opened.

    Raise StreamOpenClientError if an http client error is encountered.
    """
    from .worker import try_open_stream
    pyav_options, _ = _convert_stream_options(hass, source, options or {})
    try:
        try_open_stream(source, pyav_options).close()
    except StreamWorkerError as err:
        raise StreamOpenClientError(str(err), err.error_code) from err

def redact_credentials(url: str) -> str:
    """Redact credentials from string data."""
    yurl = URL(url)
    if yurl.user is not None:
        yurl = yurl.with_user('****')
    if yurl.password is not None:
        yurl = yurl.with_password('****')
    redacted_query_params = dict.fromkeys({'auth', 'user', 'password'} & yurl.query.keys(), '****')
    return str(yurl.update_query(redacted_query_params))

def _convert_stream_options(
    hass: HomeAssistant, stream_source: str, stream_options: Dict[str, Any]
) -> tuple[Dict[str, Any], StreamSettings]:
    """Convert options from stream options into PyAV options and stream settings."""
    if DOMAIN not in hass.data:
        raise HomeAssistantError('Stream integration is not set up.')
    stream_settings = copy.copy(hass.data[DOMAIN][ATTR_SETTINGS])
    pyav_options: Dict[str, Any] = {}
    try:
        STREAM_OPTIONS_SCHEMA(stream_options)
    except vol.Invalid as exc:
        raise HomeAssistantError(f'Invalid stream options: {exc}') from exc
    if (extra_wait_time := stream_options.get(CONF_EXTRA_PART_WAIT_TIME)):
        stream_settings.hls_part_timeout += extra_wait_time
    if (rtsp_transport := stream_options.get(CONF_RTSP_TRANSPORT)):
        assert isinstance(rtsp_transport, str)
        pyav_options['rtsp_transport'] = rtsp_transport
    if stream_options.get(CONF_USE_WALLCLOCK_AS_TIMESTAMPS):
        pyav_options['use_wallclock_as_timestamps'] = '1'
    if isinstance(stream_source, str) and stream_source[:7] == 'rtsp://':
        pyav_options = {'rtsp_flags': ATTR_PREFER_TCP, 'stimeout': '5000000', **pyav_options}
    return (pyav_options, stream_settings)

def create_stream(
    hass: HomeAssistant, stream_source: str, options: Dict[str, Any],
    dynamic_stream_settings: DynamicStreamSettings, stream_label: Optional[str] = None
) -> Stream:
    """Create a stream with the specified identifier based on the source url.

    The stream_source is typically an rtsp url (though any url accepted by ffmpeg is fine) and
    options (see STREAM_OPTIONS_SCHEMA) are converted and passed into pyav / ffmpeg.

    The stream_label is a string used as an additional message in logging.
    """
    if DOMAIN not in hass.config.components:
        raise HomeAssistantError('Stream integration is not set up.')
    pyav_options, stream_settings = _convert_stream_options(hass, stream_source, options)
    stream = Stream(
        hass, stream_source, pyav_options=pyav_options, stream_settings=stream_settings,
        dynamic_stream_settings=dynamic_stream_settings, stream_label=stream_label
    )
    hass.data[DOMAIN][ATTR_STREAMS].append(stream)
    return stream

DOMAIN_SCHEMA = vol.Schema({
    vol.Optional(CONF_LL_HLS, default=True): cv.boolean,
    vol.Optional(CONF_SEGMENT_DURATION, default=6): vol.All(cv.positive_float, vol.Range(min=2, max=10)),
    vol.Optional(CONF_PART_DURATION, default=1): vol.All(cv.positive_float, vol.Range(min=0.2, max=1.5))
})

CONFIG_SCHEMA = vol.Schema({DOMAIN: DOMAIN_SCHEMA}, extra=vol.ALLOW_EXTRA)

def set_pyav_logging(enable: bool) -> None:
    """Turn PyAV logging on or off."""
    import av
    av.logging.set_level(av.logging.VERBOSE if enable else av.logging.FATAL)

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up stream."""
    debug_enabled = _LOGGER.isEnabledFor(logging.DEBUG)

    @callback
    def update_pyav_logging(_event: Optional[Event] = None) -> None:
        """Adjust libav logging to only log when the stream logger is at DEBUG."""
        nonlocal debug_enabled
        if (new_debug_enabled := _LOGGER.isEnabledFor(logging.DEBUG)) == debug_enabled:
            return
        debug_enabled = new_debug_enabled
        set_pyav_logging(new_debug_enabled)

    cancel_logging_listener = hass.bus.async_listen(EVENT_LOGGING_CHANGED, update_pyav_logging)
    for logging_namespace in ('libav.mp4', 'libav.swscaler'):
        logging.getLogger(logging_namespace).setLevel(logging.ERROR)

    with async_pause_setup(hass, SetupPhases.WAIT_IMPORT_PACKAGES):
        await hass.async_add_executor_job(set_pyav_logging, debug_enabled)

    from .recorder import async_setup_recorder
    hass.data[DOMAIN] = {}
    hass.data[DOMAIN][ATTR_ENDPOINTS] = {}
    hass.data[DOMAIN][ATTR_STREAMS] = []
    conf = DOMAIN_SCHEMA(config.get(DOMAIN, {}))
    if conf[CONF_LL_HLS]:
        assert isinstance(conf[CONF_SEGMENT_DURATION], float)
        assert isinstance(conf[CONF_PART_DURATION], float)
        hass.data[DOMAIN][ATTR_SETTINGS] = StreamSettings(
            ll_hls=True,
            min_segment_duration=conf[CONF_SEGMENT_DURATION] - SEGMENT_DURATION_ADJUSTER,
            part_target_duration=conf[CONF_PART_DURATION],
            hls_advance_part_limit=max(int(3 / conf[CONF_PART_DURATION]), 3),
            hls_part_timeout=2 * conf[CONF_PART_DURATION]
        )
    else:
        hass.data[DOMAIN][ATTR_SETTINGS] = STREAM_SETTINGS_NON_LL_HLS

    hls_endpoint = async_setup_hls(hass)
    hass.data[DOMAIN][ATTR_ENDPOINTS][HLS_PROVIDER] = hls_endpoint
    async_setup_recorder(hass)

    async def shutdown(event: Event) -> None:
        """Stop all stream workers."""
        for stream in hass.data[DOMAIN][ATTR_STREAMS]:
            stream.dynamic_stream_settings.preload_stream = False
        if (awaitables := [create_eager_task(stream.stop()) for stream in hass.data[DOMAIN][ATTR_STREAMS]]):
            await asyncio.wait(awaitables)
        _LOGGER.debug('Stopped stream workers')
        cancel_logging_listener()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, shutdown)
    return True

class Stream:
    """Represents a single stream."""

    def __init__(
        self, hass: HomeAssistant, source: str, pyav_options: Dict[str, Any],
        stream_settings: StreamSettings, dynamic_stream_settings: DynamicStreamSettings,
        stream_label: Optional[str] = None
    ) -> None:
        """Initialize a stream."""
        self.hass = hass
        self.source = source
        self.pyav_options = pyav_options
        self._stream_settings = stream_settings
        self._stream_label = stream_label
        self.dynamic_stream_settings = dynamic_stream_settings
        self.access_token: Optional[str] = None
        self._start_stop_lock = asyncio.Lock()
        self._thread: Optional[threading.Thread] = None
        self._thread_quit = threading.Event()
        self._outputs: Dict[str, StreamOutput] = {}
        self._fast_restart_once = False
        self._keyframe_converter = KeyFrameConverter(hass, stream_settings, dynamic_stream_settings)
        self._available = True
        self._update_callback: Optional[Callable[[], None]] = None
        self._logger = logging.getLogger(f'{__package__}.stream.{stream_label}') if stream_label else _LOGGER
        self._diagnostics = Diagnostics()

    def endpoint_url(self, fmt: str) -> str:
        """Start the stream and returns a url for the output format."""
        if fmt not in self._outputs:
            raise ValueError(f"Stream is not configured for format '{fmt}'")
        if not self.access_token:
            self.access_token = secrets.token_hex()
        endpoint_fmt = self.hass.data[DOMAIN][ATTR_ENDPOINTS][fmt]
        return endpoint_fmt.format(self.access_token)

    def outputs(self) -> Mapping[str, StreamOutput]:
        """Return a copy of the stream outputs."""
        return MappingProxyType(self._outputs.copy())

    def add_provider(self, fmt: str, timeout: int = OUTPUT_IDLE_TIMEOUT) -> StreamOutput:
        """Add provider output stream."""
        if not (provider := self._outputs.get(fmt)):

            async def idle_callback() -> None:
                if (not self.dynamic_stream_settings.preload_stream or fmt == RECORDER_PROVIDER) and fmt in self._outputs:
                    await self.remove_provider(self._outputs[fmt])
                self.check_idle()

            provider = PROVIDERS[fmt](
                self.hass, IdleTimer(self.hass, timeout, idle_callback),
                self._stream_settings, self.dynamic_stream_settings
            )
            self._outputs[fmt] = provider
        return provider

    async def remove_provider(self, provider: StreamOutput) -> None:
        """Remove provider output stream."""
        if provider.name in self._outputs:
            self._outputs[provider.name].cleanup()
            del self._outputs[provider.name]
        if not self._outputs:
            await self.stop()

    def check_idle(self) -> None:
        """Reset access token if all providers are idle."""
        if all((p.idle for p in self._outputs.values())):
            self.access_token = None

    @property
    def available(self) -> bool:
        """Return False if the stream is started and known to be unavailable."""
        return self._available

    def set_update_callback(self, update_callback: Callable[[], None]) -> None:
        """Set callback to run when state changes."""
        self._update_callback = update_callback

    @callback
    def _async_update_state(self, available: bool) -> None:
        """Set state and Run callback to notify state has been updated."""
        self._available = available
        if self._update_callback:
            self._update_callback()

    async def start(self) -> None:
        """Start a stream.

        Uses an asyncio.Lock to avoid conflicts with _stop().
        """
        async with self._start_stop_lock:
            if self._thread and self._thread.is_alive():
                return
            if self._thread is not None:
                self._thread.join(timeout=0)
            self._thread_quit.clear()
            self._thread = threading.Thread(name='stream_worker', target=self._run_worker)
            self._thread.start()
            self._logger.debug('Started stream: %s', redact_credentials(str(self.source)))

    def update_source(self, new_source: str) -> None:
        """Restart the stream with a new stream source."""
        self._diagnostics.increment('update_source')
        self._logger.debug('Updating stream source %s', redact_credentials(str(new_source)))
        self.source = new_source
        self._fast_restart_once = True
        self._thread_quit.set()

    def _set_state(self, available: bool) -> None:
        """Set the stream state by updating the callback."""
        self.hass.loop.call_soon_threadsafe(self._async_update_state, available)

    def _run_worker(self) -> None:
        """Handle consuming streams and restart keepalive streams."""
        from .worker import StreamState, stream_worker
        stream_state = StreamState(self.hass, self.outputs, self._diagnostics)
        wait_timeout = 0
        while not self._thread_quit.wait(timeout=wait_timeout):
            start_time = time.time()
            self._set_state(True)
            self._diagnostics.set_value('keepalive', self.dynamic_stream_settings.preload_stream)
            self._diagnostics.set_value('orientation', self.dynamic_stream_settings.orientation)
            self._diagnostics.increment('start_worker')
            try:
                stream_worker(
                    self.source, self.pyav_options, self._stream_settings,
                    stream_state, self._keyframe_converter, self._thread_quit
                )
            except StreamWorkerError as err:
                self._diagnostics.increment('worker_error')
                self._logger.error('Error from stream worker: %s', str(err))
            stream_state.discontinuity()
            if not _should_retry() or self._thread_quit.is_set():
                if self._fast_restart_once:
                    wait_timeout = 0
                    self._fast_restart_once = False
                    self._thread_quit.clear()
                    continue
                break
            self._set_state(False)
            if time.time() - start_time > STREAM_RESTART_RESET_TIME:
                wait_timeout = 0
            wait_timeout += STREAM_RESTART_INCREMENT
            self._diagnostics.set_value('retry_timeout', wait_timeout)
            self._logger.debug('Restarting stream worker in %d seconds: %s', wait_timeout, redact_credentials(str(self.source)))

        async def worker_finished() -> None:
            if not self.available:
                self._async_update_state(True)
            for provider in self.outputs().values():
                await self.remove_provider(provider)

        self.hass.create_task(worker_finished())

    async def stop(self) -> None:
        """Remove outputs and access token."""
        self._outputs = {}
        self.access_token = None
        if not self.dynamic_stream_settings.preload_stream:
            await self._stop()

    async def _stop(self) -> None:
        """Stop worker thread.

        Uses an asyncio.Lock to avoid conflicts with start().
        """
        async with self._start_stop_lock:
            if self._thread is None:
                return
            self._thread_quit.set()
            await self.hass.async_add_executor_job(self._thread.join)
            self._thread = None
            self._logger.debug('Stopped stream: %s', redact_credentials(str(self.source)))

    async def async_record(self, video_path: str, duration: int = 30, lookback: int = 5) -> None:
        """Make a .mp4 recording from a provided stream."""
        from .recorder import RecorderOutput
        if not self.hass.config.is_allowed_path(video_path):
            raise HomeAssistantError(f"Can't write {video_path}, no access to path!")
        if (recorder := self.outputs().get(RECORDER_PROVIDER)):
            assert isinstance(recorder, RecorderOutput)
            raise HomeAssistantError(f'Stream already recording to {recorder.video_path}!')
        recorder = cast(RecorderOutput, self.add_provider(RECORDER_PROVIDER, timeout=duration))
        recorder.video_path = video_path
        await self.start()
        self._logger.debug('Started a stream recording of %s seconds', duration)
        hls = cast(HlsStreamOutput, self.outputs().get(HLS_PROVIDER))
        if hls:
            num_segments = min(int(lookback / hls.target_duration) + 1, MAX_SEGMENTS)
            await hls.recv()
            recorder.prepend(list(hls.get_segments())[-num_segments - 1:-1])
        await recorder.async_record()

    async def async_get_image(
        self, width: Optional[int] = None, height: Optional[int] = None,
        wait_for_next_keyframe: bool = False
    ) -> bytes:
        """Fetch an image from the Stream and return it as a jpeg in bytes.

        Calls async_get_image from KeyFrameConverter. async_get_image should only be
        called directly from the main loop and not from an executor thread as it uses
        hass.add_executor_job underneath the hood.
        """
        self.add_provider(HLS_PROVIDER)
        await self.start()
        return await self._keyframe_converter.async_get_image(
            width=width, height=height, wait_for_next_keyframe=wait_for_next_keyframe
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics information for the stream."""
        return self._diagnostics.as_dict()

def _should_retry() -> bool:
    """Return true if worker failures should be retried, for disabling during tests."""
    return True

STREAM_OPTIONS_SCHEMA = vol.Schema({
    vol.Optional(CONF_RTSP_TRANSPORT): vol.In(RTSP_TRANSPORTS),
    vol.Optional(CONF_USE_WALLCLOCK_AS_TIMESTAMPS): bool,
    vol.Optional(CONF_EXTRA_PART_WAIT_TIME): cv.positive_float
})
