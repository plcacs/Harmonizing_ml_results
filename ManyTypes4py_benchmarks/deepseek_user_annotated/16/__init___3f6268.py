"""Provide functionality to stream video source.

Components use create_stream with a stream source (e.g. an rtsp url) to create
a new Stream object. Stream manages:
  - Background work to fetch and decode a stream
  - Desired output formats
  - Home Assistant URLs for viewing a stream
  - Access tokens for URLs for viewing a stream

A Stream consists of a background worker, and one or more output formats each
with their own idle timeout managed by the stream component. When an output
format is no longer in use, the stream component will expire it. When there
are no active output formats, the background worker is shut down and access
tokens are expired. Alternatively, a Stream can be configured with keepalive
to always keep workers active.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import copy
import logging
import secrets
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast, Optional, Union, Dict, List, Tuple

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
    ATTR_ENDPOINTS,
    ATTR_PREFER_TCP,
    ATTR_SETTINGS,
    ATTR_STREAMS,
    CONF_EXTRA_PART_WAIT_TIME,
    CONF_LL_HLS,
    CONF_PART_DURATION,
    CONF_RTSP_TRANSPORT,
    CONF_SEGMENT_DURATION,
    CONF_USE_WALLCLOCK_AS_TIMESTAMPS,
    DOMAIN,
    FORMAT_CONTENT_TYPE,
    HLS_PROVIDER,
    MAX_SEGMENTS,
    OUTPUT_FORMATS,
    OUTPUT_IDLE_TIMEOUT,
    RECORDER_PROVIDER,
    RTSP_TRANSPORTS,
    SEGMENT_DURATION_ADJUSTER,
    SOURCE_TIMEOUT,
    STREAM_RESTART_INCREMENT,
    STREAM_RESTART_RESET_TIME,
    StreamClientError,
)
from .core import (
    PROVIDERS,
    STREAM_SETTINGS_NON_LL_HLS,
    IdleTimer,
    KeyFrameConverter,
    Orientation,
    StreamOutput,
    StreamSettings,
)
from .diagnostics import Diagnostics
from .exceptions import StreamOpenClientError, StreamWorkerError
from .hls import HlsStreamOutput, async_setup_hls

if TYPE_CHECKING:
    from homeassistant.components.camera import DynamicStreamSettings

__all__ = [
    "ATTR_SETTINGS",
    "CONF_EXTRA_PART_WAIT_TIME",
    "CONF_RTSP_TRANSPORT",
    "CONF_USE_WALLCLOCK_AS_TIMESTAMPS",
    "DOMAIN",
    "FORMAT_CONTENT_TYPE",
    "HLS_PROVIDER",
    "OUTPUT_FORMATS",
    "RTSP_TRANSPORTS",
    "SOURCE_TIMEOUT",
    "Orientation",
    "Stream",
    "StreamClientError",
    "StreamOpenClientError",
    "create_stream",
]

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


async def async_check_stream_client_error(
    hass: HomeAssistant, source: str, pyav_options: Optional[Dict[str, str]] = None
) -> None:
    """Check if a stream can be successfully opened.

    Raise StreamOpenClientError if an http client error is encountered.
    """
    await hass.loop.run_in_executor(
        None, _check_stream_client_error, hass, source, pyav_options
    )


def _check_stream_client_error(
    hass: HomeAssistant, source: str, options: Optional[Dict[str, str]] = None
) -> None:
    """Check if a stream can be successfully opened.

    Raise StreamOpenClientError if an http client error is encountered.
    """
    from .worker import try_open_stream  # pylint: disable=import-outside-toplevel

    pyav_options, _ = _convert_stream_options(hass, source, options or {})
    try:
        try_open_stream(source, pyav_options).close()
    except StreamWorkerError as err:
        raise StreamOpenClientError(str(err), err.error_code) from err


def redact_credentials(url: str) -> str:
    """Redact credentials from string data."""
    yurl = URL(url)
    if yurl.user is not None:
        yurl = yurl.with_user("****")
    if yurl.password is not None:
        yurl = yurl.with_password("****")
    redacted_query_params = dict.fromkeys(
        {"auth", "user", "password"} & yurl.query.keys(), "****"
    )
    return str(yurl.update_query(redacted_query_params))


def _convert_stream_options(
    hass: HomeAssistant,
    stream_source: str,
    stream_options: Mapping[str, Union[str, bool, float]],
) -> Tuple[Dict[str, str], StreamSettings]:
    """Convert options from stream options into PyAV options and stream settings."""
    if DOMAIN not in hass.data:
        raise HomeAssistantError("Stream integration is not set up.")

    stream_settings = copy.copy(hass.data[DOMAIN][ATTR_SETTINGS])
    pyav_options: Dict[str, str] = {}
    try:
        STREAM_OPTIONS_SCHEMA(stream_options)
    except vol.Invalid as exc:
        raise HomeAssistantError(f"Invalid stream options: {exc}") from exc

    if extra_wait_time := stream_options.get(CONF_EXTRA_PART_WAIT_TIME):
        stream_settings.hls_part_timeout += extra_wait_time
    if rtsp_transport := stream_options.get(CONF_RTSP_TRANSPORT):
        assert isinstance(rtsp_transport, str)
        # The PyAV options currently match the stream CONF constants, but this
        # will not necessarily always be the case, so they are hard coded here
        pyav_options["rtsp_transport"] = rtsp_transport
    if stream_options.get(CONF_USE_WALLCLOCK_AS_TIMESTAMPS):
        pyav_options["use_wallclock_as_timestamps"] = "1"

    # For RTSP streams, prefer TCP
    if isinstance(stream_source, str) and stream_source[:7] == "rtsp://":
        pyav_options = {
            "rtsp_flags": ATTR_PREFER_TCP,
            "stimeout": "5000000",
            **pyav_options,
        }
    return pyav_options, stream_settings


def create_stream(
    hass: HomeAssistant,
    stream_source: str,
    options: Mapping[str, Union[str, bool, float]],
    dynamic_stream_settings: DynamicStreamSettings,
    stream_label: Optional[str] = None,
) -> Stream:
    """Create a stream with the specified identifier based on the source url.

    The stream_source is typically an rtsp url (though any url accepted by ffmpeg is fine) and
    options (see STREAM_OPTIONS_SCHEMA) are converted and passed into pyav / ffmpeg.

    The stream_label is a string used as an additional message in logging.
    """

    if DOMAIN not in hass.config.components:
        raise HomeAssistantError("Stream integration is not set up.")

    # Convert extra stream options into PyAV options and stream settings
    pyav_options, stream_settings = _convert_stream_options(
        hass, stream_source, options
    )

    stream = Stream(
        hass,
        stream_source,
        pyav_options=pyav_options,
        stream_settings=stream_settings,
        dynamic_stream_settings=dynamic_stream_settings,
        stream_label=stream_label,
    )
    hass.data[DOMAIN][ATTR_STREAMS].append(stream)
    return stream


DOMAIN_SCHEMA: Final[vol.Schema] = vol.Schema(
    {
        vol.Optional(CONF_LL_HLS, default=True): cv.boolean,
        vol.Optional(CONF_SEGMENT_DURATION, default=6): vol.All(
            cv.positive_float, vol.Range(min=2, max=10)
        ),
        vol.Optional(CONF_PART_DURATION, default=1): vol.All(
            cv.positive_float, vol.Range(min=0.2, max=1.5)
        ),
    }
)

CONFIG_SCHEMA: Final[vol.Schema] = vol.Schema(
    {
        DOMAIN: DOMAIN_SCHEMA,
    },
    extra=vol.ALLOW_EXTRA,
)


def set_pyav_logging(enable: bool) -> None:
    """Turn PyAV logging on or off."""
    import av  # pylint: disable=import-outside-toplevel

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
        # enable PyAV logging iff Stream logger is set to debug
        set_pyav_logging(new_debug_enabled)

    # Only pass through PyAV log messages if stream logging is above DEBUG
    cancel_logging_listener = hass.bus.async_listen(
        EVENT_LOGGING_CHANGED, update_pyav_logging
    )
    # libav.mp4 and libav.swscaler have a few unimportant messages that are logged
    # at logging.WARNING. Set those Logger levels to logging.ERROR
    for logging_namespace in ("libav.mp4", "libav.swscaler"):
        logging.getLogger(logging_namespace).setLevel(logging.ERROR)

    # This will load av so we run it in the executor
    with async_pause_setup(hass, SetupPhases.WAIT_IMPORT_PACKAGES):
        await hass.async_add_executor_job(set_pyav_logging, debug_enabled)

    # Keep import here so that we can import stream integration without installing reqs
    # pylint: disable-next=import-outside-toplevel
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
            min_segment_duration=conf[CONF_SEGMENT_DURATION]
            - SEGMENT_DURATION_ADJUSTER,
            part_target_duration=conf[CONF_PART_DURATION],
            hls_advance_part_limit=max(int(3 / conf[CONF_PART_DURATION), 3),
            hls_part_timeout=2 * conf[CONF_PART_DURATION],
        )
    else:
        hass.data[DOMAIN][ATTR_SETTINGS] = STREAM_SETTINGS_NON_LL_HLS

    # Setup HLS
    hls_endpoint = async_setup_hls(hass)
    hass.data[DOMAIN][ATTR_ENDPOINTS][HLS_PROVIDER] = hls_endpoint

    # Setup Recorder
    async_setup_recorder(hass)

    async def shutdown(event: Event) -> None:
        """Stop all stream workers."""
        for stream in hass.data[DOMAIN][ATTR_STREAMS]:
            stream.dynamic_stream_settings.preload_stream = False
        if awaitables := [
            create_eager_task(stream.stop())
            for stream in hass.data[DOMAIN][ATTR_STREAMS]
        ]:
            await asyncio.wait(awaitables)
        _LOGGER.debug("Stopped stream workers")
        cancel_logging_listener()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, shutdown)

    return True


class Stream:
    """Represents a single stream."""

    def __init__(
        self,
        hass: HomeAssistant,
        source: str,
        pyav_options: Dict[str, str],
        stream_settings: StreamSettings,
        dynamic_stream_settings: DynamicStreamSettings,
        stream_label: Optional[str] = None,
    ) -> None:
        """Initialize a stream."""
        self.hass: HomeAssistant = hass
        self.source: str = source
        self.pyav_options: Dict[str, str] = pyav_options
        self._stream_settings: StreamSettings = stream_settings
        self._stream_label: Optional[str] = stream_label
        self.dynamic_stream_settings: DynamicStreamSettings = dynamic_stream_settings
        self.access_token: Optional[str] = None
        self._start_stop_lock: asyncio.Lock = asyncio.Lock()
        self._thread: Optional[threading.Thread] = None
        self._thread_quit: threading.Event = threading.Event()
        self._outputs: Dict[str, StreamOutput] = {}
        self._fast_restart_once: bool = False
        self._keyframe_converter: KeyFrameConverter = KeyFrameConverter(
            hass, stream_settings, dynamic_stream_settings
        )
        self._available: bool = True
        self._update_callback: Optional[Callable[[], None]] = None
        self._logger: logging.Logger = (
            logging.getLogger(f"{__package__}.stream.{stream_label}")
            if stream_label
            else _LOGGER
        )
        self._diagnostics: Diagnostics = Diagnostics()

    def endpoint_url(self, fmt: str) -> str:
        """Start the stream and returns a url for the output format."""
        if fmt not in self._outputs:
            raise ValueError(f"Stream is not configured for format '{fmt}'")
        if not self.access_token:
            self.access_token = secrets.token_hex()
        endpoint_fmt: str = self.hass.data[DOMAIN][ATTR_ENDPOINTS][fmt]
        return endpoint_fmt.format(self.access_token)

    def outputs(self) -> Mapping[str, StreamOutput]:
        """Return a copy of the stream outputs."""
        # A copy is returned so the caller can iterate through the outputs
        # without concern about self._outputs being modified from another thread.
        return MappingProxyType(self._outputs.copy())

    def add_provider(
        self, fmt: str, timeout: int = OUTPUT_IDLE_TIMEOUT
    ) -> StreamOutput:
        """Add provider output stream."""
        if not (provider := self._outputs.get(fmt)):

            async def idle_callback() -> None:
                if (
                    not self.dynamic_stream_settings.preload_stream
                    or fmt == RECORDER_PROVIDER
                ) and fmt in self._outputs:
                    await self.remove_provider(self._outputs[fmt])
                self.check_idle()

            provider = PROVIDERS[fmt](
                self.hass,
                IdleTimer(self.hass, timeout, idle_callback),
                self._stream_settings,
                self.dynamic_stream_settings,
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
        if all(p.idle for p in self._outputs.values()):
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
                # The thread must have crashed/exited. Join to clean up the
                # previous thread.
                self._thread.join(timeout=0)
            self._thread_quit.clear()
            self._thread = threading.Thread(
                name="stream_worker",
                target=self._run_worker,
            )
            self._thread.start()
            self._logger.debug(
                "Started stream: %s", redact_credentials(str(self.source))
            )

    def update_source(self, new_source: str) -> None:
        """Restart the stream with a new stream source."""
        self._diagnostics.increment("update_source")
        self._logger.debug(
            "Updating stream source %s", redact_credentials(str(new_source))
        )
        self.source = new_source
        self._fast_restart_once = True
        self._thread_quit.set()

    def _set_state(self, available: bool) -> None:
        """Set the stream state by updating the callback."""
        # Call with call_soon_threadsafe since we know _async_update_state is always
        # all callback function instead of using add_job which would have to work
        # it out each time
        self.hass.loop.call