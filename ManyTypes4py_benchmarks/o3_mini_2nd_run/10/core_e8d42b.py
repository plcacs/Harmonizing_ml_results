from __future__ import annotations
import asyncio
from collections import deque
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass, field
import datetime
from enum import IntEnum
import logging
from typing import Any, Optional, List, Tuple, TYPE_CHECKING, cast

from aiohttp import web
import numpy as np
from homeassistant.components.http import KEY_HASS, HomeAssistantView
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.util.decorator import Registry
from .const import ATTR_STREAMS, DOMAIN, SEGMENT_DURATION_ADJUSTER, TARGET_SEGMENT_DURATION_NON_LL_HLS

if TYPE_CHECKING:
    from av import Packet, VideoCodecContext
    from homeassistant.components.camera import DynamicStreamSettings
    from . import Stream

_LOGGER = logging.getLogger(__name__)
PROVIDERS = Registry()

class Orientation(IntEnum):
    """Orientations for stream transforms. These are based on EXIF orientation tags."""
    NO_TRANSFORM = 1
    MIRROR = 2
    ROTATE_180 = 3
    FLIP = 4
    ROTATE_LEFT_AND_FLIP = 5
    ROTATE_LEFT = 6
    ROTATE_RIGHT_AND_FLIP = 7
    ROTATE_RIGHT = 8

@dataclass(slots=True)
class StreamSettings:
    """Stream settings."""
    ll_hls: bool
    min_segment_duration: float
    part_target_duration: float
    hls_advance_part_limit: int
    hls_part_timeout: float

STREAM_SETTINGS_NON_LL_HLS: StreamSettings = StreamSettings(
    ll_hls=False,
    min_segment_duration=TARGET_SEGMENT_DURATION_NON_LL_HLS - SEGMENT_DURATION_ADJUSTER,
    part_target_duration=TARGET_SEGMENT_DURATION_NON_LL_HLS,
    hls_advance_part_limit=3,
    hls_part_timeout=TARGET_SEGMENT_DURATION_NON_LL_HLS
)

@dataclass(slots=True)
class Part:
    """Represent a segment part."""
    duration: float = 0.0
    data: bytes = b''  # assuming part has data attribute
    has_keyframe: bool = False  # assuming part has keyframe attribute

@dataclass(slots=True)
class Segment:
    """Represent a segment."""
    duration: float = 0.0
    parts: List[Part] = field(default_factory=list)
    hls_playlist_template: List[str] = field(default_factory=list)
    hls_playlist_parts: List[str] = field(default_factory=list)
    hls_num_parts_rendered: int = 0
    hls_playlist_complete: bool = False
    # The following attributes are assumed to be set externally:
    sequence: int = 0
    stream_id: Any = None
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    init: bytes = b''
    # This field should contain stream output objects that are updated in __post_init__
    _stream_outputs: List[StreamOutput] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Run after init."""
        for output in self._stream_outputs:
            output.put(self)

    @property
    def complete(self) -> bool:
        """Return whether the Segment is complete."""
        return self.duration > 0

    @property
    def data_size_with_init(self) -> int:
        """Return the size of all part data + init in bytes."""
        return len(self.init) + self.data_size

    @property
    def data_size(self) -> int:
        """Return the size of all part data without init in bytes."""
        return sum((len(part.data) for part in self.parts))

    @callback
    def async_add_part(self, part: Part, duration: float) -> None:
        """Add a part to the Segment.

        Duration is non zero only for the last part.
        """
        self.parts.append(part)
        self.duration = duration
        for output in self._stream_outputs:
            output.part_put()

    def get_data(self) -> bytes:
        """Return reconstructed data for all parts as bytes, without init."""
        return b''.join([part.data for part in self.parts])

    def _render_hls_template(self, last_stream_id: Any, render_parts: bool) -> str:
        """Render the HLS playlist section for the Segment.

        The Segment may still be in progress.
        This method stores intermediate data in hls_playlist_parts,
        hls_num_parts_rendered, and hls_playlist_complete to avoid redoing
        work on subsequent calls.
        """
        if self.hls_playlist_complete:
            return self.hls_playlist_template[0]
        if not self.hls_playlist_template:
            if last_stream_id != self.stream_id:
                self.hls_playlist_template.append('#EXT-X-DISCONTINUITY')
            self.hls_playlist_template.append('{}')
        if render_parts:
            for part_num, part in enumerate(self.parts[self.hls_num_parts_rendered:], self.hls_num_parts_rendered):
                self.hls_playlist_parts.append(
                    f'#EXT-X-PART:DURATION={part.duration:.3f},URI="./segment/{self.sequence}.{part_num}.m4s"{(",INDEPENDENT=YES" if part.has_keyframe else "")}'
                )
        if self.complete:
            self.hls_playlist_parts.append('')
            if last_stream_id == self.stream_id:
                self.hls_playlist_template = []
            else:
                self.hls_playlist_template = ['#EXT-X-DISCONTINUITY']
            self.hls_playlist_template.extend([
                '{}#EXT-X-PROGRAM-DATE-TIME:' + self.start_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                f'#EXTINF:{self.duration:.3f},\n./segment/{self.sequence}.m4s'
            ])
        self.hls_playlist_template = ['\n'.join(self.hls_playlist_template)]
        self.hls_playlist_parts = ['\n'.join(self.hls_playlist_parts).lstrip()]
        self.hls_num_parts_rendered = len(self.parts)
        self.hls_playlist_complete = self.complete
        return self.hls_playlist_template[0]

    def render_hls(self, last_stream_id: Any, render_parts: bool, add_hint: bool) -> str:
        """Render the HLS playlist section for the Segment including a hint if requested."""
        playlist_template: str = self._render_hls_template(last_stream_id, render_parts)
        playlist: str = playlist_template.format(self.hls_playlist_parts[0] if render_parts else '')
        if not add_hint:
            return playlist
        if self.complete:
            sequence: int = self.sequence + 1
            part_num: int = 0
        else:
            sequence = self.sequence
            part_num = len(self.parts)
        hint: str = f'#EXT-X-PRELOAD-HINT:TYPE=PART,URI="./segment/{sequence}.{part_num}.m4s"'
        return (playlist + '\n' + hint) if playlist else hint

class IdleTimer:
    """Invoke a callback after an inactivity timeout.

    The IdleTimer invokes the callback after some timeout has passed. The awake() method
    resets the internal alarm, extending the inactivity time.
    """

    def __init__(self, hass: HomeAssistant, timeout: float, idle_callback: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Initialize IdleTimer."""
        self._hass: HomeAssistant = hass
        self._timeout: float = timeout
        self._callback: Callable[[], Coroutine[Any, Any, Any]] = idle_callback
        self._unsub: Optional[CALLBACK_TYPE] = None
        self.idle: bool = False

    def start(self) -> None:
        """Start the idle timer if not already started."""
        self.idle = False
        if self._unsub is None:
            self._unsub = async_call_later(self._hass, self._timeout, self.fire)

    def awake(self) -> None:
        """Keep the idle time alive by resetting the timeout."""
        self.idle = False
        self.clear()
        self._unsub = async_call_later(self._hass, self._timeout, self.fire)

    def clear(self) -> None:
        """Clear and disable the timer if it has not already fired."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    @callback
    def fire(self, _now: Any) -> None:
        """Invoke the idle timeout callback, called when the alarm fires."""
        self.idle = True
        self._unsub = None
        self._hass.async_create_task(self._callback())

class StreamOutput:
    """Represents a stream output."""

    def __init__(
        self,
        hass: HomeAssistant,
        idle_timer: IdleTimer,
        stream_settings: StreamSettings,
        dynamic_stream_settings: DynamicStreamSettings,
        deque_maxlen: Optional[int] = None
    ) -> None:
        """Initialize a stream output."""
        self._hass: HomeAssistant = hass
        self.idle_timer: IdleTimer = idle_timer
        self.stream_settings: StreamSettings = stream_settings
        self.dynamic_stream_settings: DynamicStreamSettings = dynamic_stream_settings
        self._event: asyncio.Event = asyncio.Event()
        self._part_event: asyncio.Event = asyncio.Event()
        self._segments: deque[Segment] = deque(maxlen=deque_maxlen)

    @property
    def name(self) -> Optional[str]:
        """Return provider name."""
        return None

    @property
    def idle(self) -> bool:
        """Return True if the output is idle."""
        return self.idle_timer.idle

    @property
    def last_sequence(self) -> int:
        """Return the last sequence number without iterating."""
        if self._segments:
            return self._segments[-1].sequence
        return -1

    @property
    def sequences(self) -> List[int]:
        """Return current sequence from segments."""
        return [s.sequence for s in self._segments]

    @property
    def last_segment(self) -> Optional[Segment]:
        """Return the last segment without iterating."""
        if self._segments:
            return self._segments[-1]
        return None

    def get_segment(self, sequence: int) -> Optional[Segment]:
        """Retrieve a specific segment."""
        for segment in reversed(self._segments):
            if segment.sequence == sequence:
                return segment
        return None

    def get_segments(self) -> deque[Segment]:
        """Retrieve all segments."""
        return self._segments

    async def part_recv(self, timeout: Optional[float] = None) -> bool:
        """Wait for an event signalling the latest part segment."""
        try:
            async with asyncio.timeout(timeout):
                await self._part_event.wait()
        except TimeoutError:
            return False
        return True

    def part_put(self) -> None:
        """Set event signalling the latest part segment."""
        self._part_event.set()
        self._part_event.clear()

    async def recv(self) -> bool:
        """Wait for the latest segment."""
        await self._event.wait()
        return self.last_segment is not None

    def put(self, segment: Segment) -> None:
        """Store output."""
        self._hass.loop.call_soon_threadsafe(self._async_put, segment)

    @callback
    def _async_put(self, segment: Segment) -> None:
        """Store output from event loop."""
        self.idle_timer.start()
        self._segments.append(segment)
        self._event.set()
        self._event.clear()

    def cleanup(self) -> None:
        """Handle cleanup."""
        self._event.set()
        self.idle_timer.clear()

class StreamView(HomeAssistantView):
    """Base StreamView.

    For implementation of a new stream format, define `url` and `name`
    attributes, and implement `handle` method in a child class.
    """
    requires_auth = False

    async def get(self, request: web.Request, token: str, sequence: str = '', part_num: str = '') -> web.Response:
        """Start a GET request."""
        hass: HomeAssistant = request.app[KEY_HASS]
        stream: Optional[Stream] = next(
            (s for s in hass.data[DOMAIN][ATTR_STREAMS] if s.access_token == token), None
        )
        if not stream:
            raise web.HTTPNotFound
        await stream.start()
        return await self.handle(request, stream, sequence, part_num)

    async def handle(self, request: web.Request, stream: Stream, sequence: str, part_num: str) -> web.Response:
        """Handle the stream request."""
        raise NotImplementedError

TRANSFORM_IMAGE_FUNCTION: Tuple[Callable[[np.ndarray], np.ndarray], ...] = (
    lambda image: image,
    lambda image: image,
    lambda image: np.fliplr(image).copy(),
    lambda image: np.rot90(image, 2).copy(),
    lambda image: np.flipud(image).copy(),
    lambda image: np.flipud(np.rot90(image)).copy(),
    lambda image: np.rot90(image).copy(),
    lambda image: np.flipud(np.rot90(image, -1)).copy(),
    lambda image: np.rot90(image, -1).copy()
)

class KeyFrameConverter:
    """Enables generating and getting an image from the last keyframe seen in the stream.

    An overview of the thread and state interaction:
        the worker thread sets a packet
        get_image is called from the main asyncio loop
        get_image schedules _generate_image in an executor thread
        _generate_image will try to create an image from the packet
        _generate_image will clear the packet, so there will only be one attempt per packet
    If successful, self._image will be updated and returned by get_image
    If unsuccessful, get_image will return the previous image
    """

    def __init__(self, hass: HomeAssistant, stream_settings: StreamSettings, dynamic_stream_settings: DynamicStreamSettings) -> None:
        """Initialize."""
        from homeassistant.components.camera.img_util import TurboJPEGSingleton
        self._packet: Optional[Packet] = None
        self._event: asyncio.Event = asyncio.Event()
        self._hass: HomeAssistant = hass
        self._image: Optional[bytes] = None
        self._turbojpeg = TurboJPEGSingleton.instance()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._codec_context: Optional[VideoCodecContext] = None
        self._stream_settings: StreamSettings = stream_settings
        self._dynamic_stream_settings: DynamicStreamSettings = dynamic_stream_settings

    def stash_keyframe_packet(self, packet: Packet) -> None:
        """Store the keyframe and set the asyncio.Event from the event loop.

        This is called from the worker thread.
        """
        self._packet = packet
        self._hass.loop.call_soon_threadsafe(self._event.set)

    def create_codec_context(self, codec_context: VideoCodecContext) -> None:
        """Create a codec context to be used for decoding the keyframes.

        This is run by the worker thread and will only be called once per worker.
        """
        if self._codec_context:
            return
        from av import CodecContext
        self._codec_context = cast(VideoCodecContext, CodecContext.create(codec_context.name, 'r'))
        self._codec_context.extradata = codec_context.extradata
        self._codec_context.skip_frame = 'NONKEY'
        self._codec_context.thread_type = 'NONE'

    @staticmethod
    def transform_image(image: np.ndarray, orientation: int) -> np.ndarray:
        """Transform image to a given orientation."""
        return TRANSFORM_IMAGE_FUNCTION[orientation](image)

    def _generate_image(self, width: Optional[int], height: Optional[int]) -> None:
        """Generate the keyframe image.

        This is run in an executor thread, but since it is called within an
        the asyncio lock from the main thread, there will only be one entry
        at a time per instance.
        """
        if not (self._turbojpeg and self._packet and self._codec_context):
            return
        packet: Packet = self._packet
        self._packet = None
        for _ in range(2):
            try:
                frames = self._codec_context.decode(packet)
                for _i in range(2):
                    if frames:
                        break
                    frames = self._codec_context.decode(None)
                break
            except EOFError:
                _LOGGER.debug('Codec context needs flushing')
                self._codec_context.flush_buffers()
        else:
            _LOGGER.debug('Unable to decode keyframe')
            return
        if frames:
            frame = frames[0]
            if width and height:
                if self._dynamic_stream_settings.orientation >= 5:
                    frame = frame.reformat(width=height, height=width)
                else:
                    frame = frame.reformat(width=width, height=height)
            bgr_array = self.transform_image(frame.to_ndarray(format='bgr24'), self._dynamic_stream_settings.orientation)
            self._image = bytes(self._turbojpeg.encode(bgr_array))

    async def async_get_image(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        wait_for_next_keyframe: bool = False
    ) -> Optional[bytes]:
        """Fetch an image from the Stream and return it as a jpeg in bytes."""
        if wait_for_next_keyframe:
            self._event.clear()
            await self._event.wait()
        async with self._lock:
            await self._hass.async_add_executor_job(self._generate_image, width, height)
        return self._image