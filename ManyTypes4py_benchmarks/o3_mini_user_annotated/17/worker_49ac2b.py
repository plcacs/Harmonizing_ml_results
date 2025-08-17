"""Provides the worker thread needed for processing streams."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterator, Mapping
import contextlib
from dataclasses import fields
import datetime
from io import SEEK_END, BytesIO
import logging
from threading import Event
from typing import Any, Optional, Self, cast

import av
import av.audio
import av.container
from av.container import InputContainer
import av.stream

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from . import redact_credentials
from .const import (
    AUDIO_CODECS,
    HLS_PROVIDER,
    MAX_MISSING_DTS,
    MAX_TIMESTAMP_GAP,
    PACKETS_TO_WAIT_FOR_AUDIO,
    SEGMENT_CONTAINER_FORMAT,
    SOURCE_TIMEOUT,
    StreamClientError,
)
from .core import (
    STREAM_SETTINGS_NON_LL_HLS,
    KeyFrameConverter,
    Part,
    Segment,
    StreamOutput,
    StreamSettings,
)
from .diagnostics import Diagnostics
from .exceptions import StreamEndedError, StreamWorkerError
from .fmp4utils import read_init
from .hls import HlsStreamOutput

_LOGGER = logging.getLogger(__name__)
NEGATIVE_INF = float("-inf")


def redact_av_error_string(err: av.FFmpegError) -> str:
    """Return an error string with credentials redacted from the url."""
    parts: list[str] = [str(err.type), err.strerror]  # type: ignore[attr-defined]
    if err.filename:
        parts.append(redact_credentials(err.filename))
    return ", ".join(parts)


class StreamState:
    """Responsible for tracking output and playback state for a stream.

    Holds state used for playback to interpret a decoded stream. A source stream
    may be reset (e.g. reconnecting to an rtsp stream) and this object tracks
    the state to inform the player.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        outputs_callback: Callable[[], Mapping[str, StreamOutput]],
        diagnostics: Diagnostics,
    ) -> None:
        """Initialize StreamState."""
        self._stream_id: int = 0
        self.hass: HomeAssistant = hass
        self._outputs_callback: Callable[[], Mapping[str, StreamOutput]] = outputs_callback
        # sequence gets incremented before the first segment so the first segment
        # has a sequence number of 0.
        self._sequence: int = -1
        self._diagnostics: Diagnostics = diagnostics

    @property
    def sequence(self) -> int:
        """Return the current sequence for the latest segment."""
        return self._sequence

    def next_sequence(self) -> int:
        """Increment the sequence number."""
        self._sequence += 1
        return self._sequence

    @property
    def stream_id(self) -> int:
        """Return the readonly stream_id attribute."""
        return self._stream_id

    def discontinuity(self) -> None:
        """Mark the stream as having been restarted."""
        # Preserving sequence and stream_id here keep the HLS playlist logic
        # simple to check for discontinuity at output time, and to determine
        # the discontinuity sequence number.
        self._stream_id += 1
        # Call discontinuity to fix incomplete segment in HLS output
        if hls_output := self._outputs_callback().get(HLS_PROVIDER):
            cast(HlsStreamOutput, hls_output).discontinuity()

    @property
    def outputs(self) -> list[StreamOutput]:
        """Return the active stream outputs."""
        return list(self._outputs_callback().values())

    @property
    def diagnostics(self) -> Diagnostics:
        """Return diagnostics object."""
        return self._diagnostics


class StreamMuxer:
    """StreamMuxer re-packages video/audio packets for output."""

    _segment_start_dts: int
    _memory_file: BytesIO
    _av_output: av.container.OutputContainer
    _output_video_stream: av.VideoStream
    _output_audio_stream: Optional[av.audio.AudioStream]
    _segment: Optional[Segment]
    # the following 2 member variables are used for Part formation
    _memory_file_pos: int
    _part_start_dts: float

    def __init__(
        self,
        hass: HomeAssistant,
        video_stream: av.VideoStream,
        audio_stream: Optional[av.audio.AudioStream],
        audio_bsf: Optional[str],
        stream_state: StreamState,
        stream_settings: StreamSettings,
    ) -> None:
        """Initialize StreamMuxer."""
        self._hass: HomeAssistant = hass
        self._input_video_stream: av.VideoStream = video_stream
        self._input_audio_stream: Optional[av.audio.AudioStream] = audio_stream
        self._audio_bsf: Optional[str] = audio_bsf
        self._audio_bsf_context: Optional[av.BitStreamFilterContext] = None
        self._part_has_keyframe: bool = False
        self._stream_settings: StreamSettings = stream_settings
        self._stream_state: StreamState = stream_state
        self._start_time: datetime.datetime = dt_util.utcnow()

    def make_new_av(
        self,
        memory_file: BytesIO,
        sequence: int,
        input_vstream: av.VideoStream,
        input_astream: Optional[av.audio.AudioStream],
    ) -> tuple[av.container.OutputContainer, av.VideoStream, Optional[av.audio.AudioStream]]:
        """Make a new av OutputContainer and add output streams."""
        container_options: dict[str, str] = {
            # Removed skip_sidx - see:
            # https://github.com/home-assistant/core/pull/39970
            # "cmaf" flag replaces several of the movflags used,
            # but too recent to use for now
            "movflags": "frag_custom+empty_moov+default_base_moof+frag_discont+negative_cts_offsets+skip_trailer+delay_moov",
            # Sometimes the first segment begins with negative timestamps,
            # and this setting just
            # adjusts the timestamps in the output from that segment to start
            # from 0. Helps from having to make some adjustments
            # in test_durations
            "avoid_negative_ts": "make_non_negative",
            "fragment_index": str(sequence + 1),
            "video_track_timescale": str(int(1 / input_vstream.time_base)),  # type: ignore[operator]
            # Only do extra fragmenting if we are using ll_hls
            # Let ffmpeg do the work using frag_duration
            # Fragment durations may exceed the 15% allowed variance but it seems ok
            **(
                {
                    "movflags": "empty_moov+default_base_moof+frag_discont+negative_cts_offsets+skip_trailer+delay_moov",
                    # Create a fragment every TARGET_PART_DURATION. The data from
                    # each fragment is stored in a "Part" that can be combined with
                    # the data from all the other "Part"s, plus an init section,
                    # to reconstitute the data in a "Segment".
                    #
                    # The LL-HLS spec allows for a fragment's duration to be within
                    # the range [0.85x,1.0x] of the part target duration. We use the
                    # frag_duration option to tell ffmpeg to try to cut the
                    # fragments when they reach frag_duration. However,
                    # the resulting fragments can have variability in their
                    # durations and can end up being too short or too long. With a
                    # video track with no audio, the discrete nature of frames means
                    # that the frame at the end of a fragment will sometimes extend
                    # slightly beyond the desired frag_duration.
                    #
                    # If there are two tracks, as in the case of a video feed with
                    # audio, there is an added wrinkle as the fragment cut seems to
                    # be done on the first track that crosses the desired threshold,
                    # and cutting on the audio track may also result in a shorter
                    # video fragment than desired.
                    #
                    # Given this, our approach is to give ffmpeg a frag_duration
                    # somewhere in the middle of the range, hoping that the parts
                    # stay pretty well bounded, and we adjust the part durations
                    # a bit in the hls metadata so that everything "looks" ok.
                    "frag_duration": str(
                        int(self._stream_settings.part_target_duration * 9e5)
                    ),
                }
                if self._stream_settings.ll_hls
                else {}
            ),
        }
        container: av.container.OutputContainer = av.open(
            memory_file,
            mode="w",
            format=SEGMENT_CONTAINER_FORMAT,
            container_options=container_options,
        )
        output_vstream: av.VideoStream = container.add_stream(template=input_vstream)
        output_astream: Optional[av.audio.AudioStream] = None
        if input_astream:
            if self._audio_bsf:
                self._audio_bsf_context = av.BitStreamFilterContext(
                    self._audio_bsf, input_astream
                )
            output_astream = container.add_stream(template=input_astream)
        return container, output_vstream, output_astream  # type: ignore[return-value]

    def reset(self, video_dts: int) -> None:
        """Initialize a new stream segment."""
        self._part_start_dts = self._segment_start_dts = video_dts
        self._segment = None
        self._memory_file = BytesIO()
        self._memory_file_pos = 0
        (
            self._av_output,
            self._output_video_stream,
            self._output_audio_stream,
        ) = self.make_new_av(
            memory_file=self._memory_file,
            sequence=self._stream_state.next_sequence(),
            input_vstream=self._input_video_stream,
            input_astream=self._input_audio_stream,
        )
        if self._output_video_stream.name == "hevc":
            self._output_video_stream.codec_context.codec_tag = "hvc1"

    def mux_packet(self, packet: av.Packet) -> None:
        """Mux a packet to the appropriate output stream."""
        if packet.stream == self._input_video_stream:
            if (
                packet.is_keyframe
                and (packet.dts - self._segment_start_dts) * packet.time_base
                >= self._stream_settings.min_segment_duration
            ):
                self.flush(packet, last_part=True)
            packet.stream = self._output_video_stream
            self._av_output.mux(packet)
            self.check_flush_part(packet)
            self._part_has_keyframe |= packet.is_keyframe
        elif packet.stream == self._input_audio_stream:
            assert self._output_audio_stream is not None
            if self._audio_bsf_context:
                for audio_packet in self._audio_bsf_context.filter(packet):
                    audio_packet.stream = self._output_audio_stream
                    self._av_output.mux(audio_packet)
                return
            packet.stream = self._output_audio_stream
            self._av_output.mux(packet)

    def create_segment(self) -> None:
        """Create a segment when the moov is ready."""
        self._segment = Segment(
            sequence=self._stream_state.sequence,
            stream_id=self._stream_state.stream_id,
            init=read_init(self._memory_file),
            _stream_outputs=self._stream_state.outputs,
            start_time=self._start_time,
        )
        self._memory_file_pos = self._memory_file.tell()
        self._memory_file.seek(0, SEEK_END)

    def check_flush_part(self, packet: av.Packet) -> None:
        """Check for and mark a part segment boundary and record its duration."""
        if self._memory_file_pos == self._memory_file.tell():
            return
        if self._segment is None:
            self.create_segment()
            self.flush(packet, last_part=False)
        else:
            self.flush(packet, last_part=False)

    def flush(self, packet: av.Packet, last_part: bool) -> None:
        """Output a part from the most recent bytes in the memory_file.

        If last_part is True, also close the segment, give it a duration,
        and clean up the av_output and memory_file.
        There are two different ways to enter this function, and when
        last_part is True, packet has not yet been muxed, while when
        last_part is False, the packet has already been muxed. However,
        in both cases, packet is the next packet and is not included in
        the Part.
        This function writes the duration metadata for the Part and
        for the Segment. However, as the fragmentation done by ffmpeg
        may result in fragment durations which fall outside the
        [0.85x,1.0x] tolerance band allowed by LL-HLS, we need to fudge
        some durations a bit by reporting them as being within that
        range.
        Note that repeated adjustments may cause drift between the part
        durations in the metadata and those in the media and result in
        playback issues in some clients.
        """
        adjusted_dts: int = min(
            packet.dts,
            self._part_start_dts
            + self._stream_settings.part_target_duration / packet.time_base,
        )
        if last_part:
            self._av_output.close()
            if not self._segment:
                self.create_segment()
        elif not self._part_has_keyframe:
            adjusted_dts = max(
                adjusted_dts,
                self._part_start_dts
                + 0.85 * self._stream_settings.part_target_duration / packet.time_base,
            )
        if not self._stream_settings.ll_hls:
            adjusted_dts = packet.dts
        assert self._segment is not None
        self._memory_file.seek(self._memory_file_pos)
        segment_duration: float = 0
        self._hass.loop.call_soon_threadsafe(
            self._segment.async_add_part,
            Part(
                duration=float((adjusted_dts - self._part_start_dts) * packet.time_base),
                has_keyframe=self._part_has_keyframe,
                data=self._memory_file.read(),
            ),
            (
                segment_duration
                if last_part
                else 0
            ),
        )
        if last_part:
            self._memory_file.close()
            segment_duration = float((adjusted_dts - self._segment_start_dts) * packet.time_base)
            self._start_time += datetime.timedelta(seconds=segment_duration)
            self.reset(packet.dts)
        else:
            self._memory_file_pos = self._memory_file.tell()
            self._part_start_dts = adjusted_dts
        self._part_has_keyframe = False

    def close(self) -> None:
        """Close stream buffer."""
        self._av_output.close()
        self._memory_file.close()


class PeekIterator(Iterator[av.Packet]):
    """An Iterator that may allow multiple passes.

    This may be consumed like a normal Iterator, however also supports a
    peek() method that buffers consumed items from the iterator.
    """

    def __init__(self, iterator: Iterator[av.Packet]) -> None:
        """Initialize PeekIterator."""
        self._iterator: Iterator[av.Packet] = iterator
        self._buffer: deque[av.Packet] = deque()
        self._next: Callable[[], av.Packet] = self._iterator.__next__

    def __iter__(self) -> Self:
        """Return an iterator."""
        return self

    def __next__(self) -> av.Packet:
        """Return and consume the next item available."""
        return self._next()

    def _pop_buffer(self) -> av.Packet:
        """Consume items from the buffer until exhausted."""
        if self._buffer:
            return self._buffer.popleft()
        self._next = self._iterator.__next__
        return self._next()

    def peek(self) -> Generator[av.Packet, None, None]:
        """Return items without consuming from the iterator."""
        self._next = self._pop_buffer
        yield from self._buffer
        for packet in self._iterator:
            self._buffer.append(packet)
            yield packet


class TimestampValidator:
    """Validate ordering of timestamps for packets in a stream."""

    def __init__(self, inv_video_time_base: int, inv_audio_time_base: int) -> None:
        """Initialize the TimestampValidator."""
        self._last_dts: dict[av.stream.Stream, int | float] = defaultdict(
            lambda: NEGATIVE_INF
        )
        self._missing_dts: int = 0
        self._max_dts_gap: int = MAX_TIMESTAMP_GAP * max(
            inv_video_time_base, inv_audio_time_base
        )

    def is_valid(self, packet: av.Packet) -> bool:
        """Validate the packet timestamp based on ordering within the stream."""
        if packet.dts is None:
            if self._missing_dts >= MAX_MISSING_DTS:
                raise StreamWorkerError(
                    f"No dts in {MAX_MISSING_DTS + 1} consecutive packets"
                )
            self._missing_dts += 1
            return False
        self._missing_dts = 0
        prev_dts: int | float = self._last_dts[packet.stream]
        if abs(prev_dts - packet.dts) > self._max_dts_gap and prev_dts != NEGATIVE_INF:
            raise StreamWorkerError(
                f"Timestamp discontinuity detected: last dts = {prev_dts}, dts = {packet.dts}"
            )
        if packet.dts <= prev_dts:
            return False
        self._last_dts[packet.stream] = packet.dts
        return True


def is_keyframe(packet: av.Packet) -> Any:
    """Return true if the packet is a keyframe."""
    return packet.is_keyframe


def get_audio_bitstream_filter(
    packets: Iterator[av.Packet], audio_stream: Any
) -> Optional[str]:
    """Return the aac_adtstoasc bitstream filter if ADTS AAC is detected."""
    if not audio_stream:
        return None
    for count, packet in enumerate(packets):
        if count >= PACKETS_TO_WAIT_FOR_AUDIO:
            _LOGGER.warning("Audio stream not found")
            break
        if packet.stream == audio_stream:
            if audio_stream.codec.name == "aac" and packet.size > 2:
                with memoryview(packet) as packet_view:
                    if packet_view[0] == 0xFF and packet_view[1] & 0xF0 == 0xF0:
                        _LOGGER.debug(
                            "ADTS AAC detected. Adding aac_adtstoaac bitstream filter"
                        )
                        return "aac_adtstoasc"
            break
    return None


def try_open_stream(
    source: str,
    pyav_options: dict[str, str],
) -> InputContainer:
    """Try to open a stream.

    Will raise StreamOpenClientError if an http client error is encountered.
    """
    try:
        return av.open(source, options=pyav_options, timeout=SOURCE_TIMEOUT)
    except av.HTTPBadRequestError as err:
        raise StreamWorkerError(
            f"Bad Request Error opening stream ({redact_av_error_string(err)})",
            error_code=StreamClientError.BadRequest,
        ) from err

    except av.HTTPUnauthorizedError as err:
        raise StreamWorkerError(
            f"Unauthorized error opening stream ({redact_av_error_string(err)})",
            error_code=StreamClientError.Unauthorized,
        ) from err

    except av.HTTPForbiddenError as err:
        raise StreamWorkerError(
            f"Forbidden error opening stream ({redact_av_error_string(err)})",
            error_code=StreamClientError.Forbidden,
        ) from err

    except av.HTTPNotFoundError as err:
        raise StreamWorkerError(
            f"Not Found error opening stream ({redact_av_error_string(err)})",
            error_code=StreamClientError.NotFound,
        ) from err

    except av.FFmpegError as err:
        raise StreamWorkerError(
            f"Error opening stream ({redact_av_error_string(err)})"
        ) from err


def stream_worker(
    source: str,
    pyav_options: dict[str, str],
    stream_settings: StreamSettings,
    stream_state: StreamState,
    keyframe_converter: KeyFrameConverter,
    quit_event: Event,
) -> None:
    """Handle consuming streams."""
    if av.library_versions["libavformat"][0] >= 59 and "stimeout" in pyav_options:
        pyav_options["timeout"] = pyav_options["stimeout"]
        del pyav_options["stimeout"]
    container: InputContainer = try_open_stream(source, pyav_options)
    try:
        video_stream: av.VideoStream = container.streams.video[0]
    except (KeyError, IndexError) as ex:
        raise StreamWorkerError("Stream has no video") from ex
    keyframe_converter.create_codec_context(codec_context=video_stream.codec_context)
    try:
        audio_stream: Optional[av.audio.AudioStream] = container.streams.audio[0]
    except (KeyError, IndexError):
        audio_stream = None
    if audio_stream and audio_stream.name not in AUDIO_CODECS:
        audio_stream = None
    if audio_stream and audio_stream.profile is None:
        audio_stream = None  # type: ignore[unreachable]
    if container.format.name == "hls":
        for field in fields(StreamSettings):
            setattr(
                stream_settings,
                field.name,
                getattr(STREAM_SETTINGS_NON_LL_HLS, field.name),
            )
    stream_state.diagnostics.set_value("container_format", container.format.name)
    stream_state.diagnostics.set_value("video_codec", video_stream.name)
    if audio_stream:
        stream_state.diagnostics.set_value("audio_codec", audio_stream.name)

    dts_validator: TimestampValidator = TimestampValidator(
        int(1 / video_stream.time_base),  # type: ignore[operator]
        int(1 / audio_stream.time_base) if audio_stream else 1,  # type: ignore[operator]
    )
    container_packets: PeekIterator = PeekIterator(
        filter(dts_validator.is_valid, container.demux((video_stream, audio_stream)))
    )

    def is_video(packet: av.Packet) -> Any:
        """Return true if the packet is for the video stream."""
        return packet.stream.type == "video"

    try:
        audio_bsf: Optional[str] = get_audio_bitstream_filter(container_packets.peek(), audio_stream)
        first_keyframe: av.Packet = next(
            filter(lambda pkt: is_keyframe(pkt) and is_video(pkt), container_packets)
        )
        next_video_packet: av.Packet = next(filter(is_video, container_packets.peek()))
        start_dts: int = next_video_packet.dts - (next_video_packet.duration or 1)
        first_keyframe.dts = first_keyframe.pts = start_dts
    except StreamWorkerError:
        container.close()
        raise
    except StopIteration as ex:
        container.close()
        raise StreamEndedError("Stream ended; no additional packets") from ex
    except av.FFmpegError as ex:
        container.close()
        raise StreamWorkerError(
            f"Error demuxing stream while finding first packet ({redact_av_error_string(ex)})"
        ) from ex

    muxer: StreamMuxer = StreamMuxer(
        stream_state.hass,
        video_stream,
        audio_stream,
        audio_bsf,
        stream_state,
        stream_settings,
    )
    muxer.reset(start_dts)
    muxer.mux_packet(first_keyframe)

    with contextlib.closing(container), contextlib.closing(muxer):
        while not quit_event.is_set():
            try:
                packet: av.Packet = next(container_packets)
            except StreamWorkerError:
                raise
            except StopIteration as ex:
                raise StreamEndedError("Stream ended; no additional packets") from ex
            except av.FFmpegError as ex:
                raise StreamWorkerError(
                    f"Error demuxing stream ({redact_av_error_string(ex)})"
                ) from ex

            muxer.mux_packet(packet)

            if packet.is_keyframe and is_video(packet):
                keyframe_converter.stash_keyframe_packet(packet)