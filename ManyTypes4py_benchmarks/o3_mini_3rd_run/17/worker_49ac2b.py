from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Generator, Iterator, Mapping
import contextlib
from dataclasses import fields
import datetime
from io import SEEK_END, BytesIO
import logging
from threading import Event
from typing import Any, Deque, Optional, Self, cast

import av
import av.audio
import av.container
from av.container import InputContainer
import av.stream
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from . import redact_credentials
from .const import AUDIO_CODECS, HLS_PROVIDER, MAX_MISSING_DTS, MAX_TIMESTAMP_GAP, PACKETS_TO_WAIT_FOR_AUDIO, SEGMENT_CONTAINER_FORMAT, SOURCE_TIMEOUT, StreamClientError
from .core import STREAM_SETTINGS_NON_LL_HLS, KeyFrameConverter, Part, Segment, StreamOutput, StreamSettings
from .diagnostics import Diagnostics
from .exceptions import StreamEndedError, StreamWorkerError
from .fmp4utils import read_init
from .hls import HlsStreamOutput

_LOGGER = logging.getLogger(__name__)
NEGATIVE_INF = float('-inf')


def redact_av_error_string(err: av.FFmpegError) -> str:
    parts = [str(err.type), err.strerror]
    if err.filename:
        parts.append(redact_credentials(err.filename))
    return ', '.join(parts)


class StreamState:
    def __init__(
        self,
        hass: HomeAssistant,
        outputs_callback: Callable[[], Mapping[str, StreamOutput]],
        diagnostics: Diagnostics,
    ) -> None:
        self._stream_id: int = 0
        self.hass: HomeAssistant = hass
        self._outputs_callback: Callable[[], Mapping[str, StreamOutput]] = outputs_callback
        self._sequence: int = -1
        self._diagnostics: Diagnostics = diagnostics

    @property
    def sequence(self) -> int:
        return self._sequence

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    @property
    def stream_id(self) -> int:
        return self._stream_id

    def discontinuity(self) -> None:
        self._stream_id += 1
        if (hls_output := self._outputs_callback().get(HLS_PROVIDER)):
            cast(HlsStreamOutput, hls_output).discontinuity()

    @property
    def outputs(self) -> list[StreamOutput]:
        return list(self._outputs_callback().values())

    @property
    def diagnostics(self) -> Diagnostics:
        return self._diagnostics


class StreamMuxer:
    def __init__(
        self,
        hass: HomeAssistant,
        video_stream: av.stream.Stream,
        audio_stream: Optional[av.stream.Stream],
        audio_bsf: Optional[str],
        stream_state: StreamState,
        stream_settings: StreamSettings,
    ) -> None:
        self._hass: HomeAssistant = hass
        self._input_video_stream: av.stream.Stream = video_stream
        self._input_audio_stream: Optional[av.stream.Stream] = audio_stream
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
        input_vstream: av.stream.Stream,
        input_astream: Optional[av.stream.Stream],
    ) -> tuple[av.container.OutputContainer, av.stream.Stream, Optional[av.stream.Stream]]:
        container_options: dict[str, Any] = {
            'movflags': 'frag_custom+empty_moov+default_base_moof+frag_discont+negative_cts_offsets+skip_trailer+delay_moov',
            'avoid_negative_ts': 'make_non_negative',
            'fragment_index': str(sequence + 1),
            'video_track_timescale': str(int(1 / input_vstream.time_base)),
            **(
                {
                    'movflags': 'empty_moov+default_base_moof+frag_discont+negative_cts_offsets+skip_trailer+delay_moov',
                    'frag_duration': str(int(self._stream_settings.part_target_duration * 900000.0)),
                }
                if self._stream_settings.ll_hls
                else {}
            ),
        }
        container: av.container.OutputContainer = av.open(memory_file, mode='w', format=SEGMENT_CONTAINER_FORMAT, container_options=container_options)
        output_vstream: av.stream.Stream = container.add_stream(template=input_vstream)
        output_astream: Optional[av.stream.Stream] = None
        if input_astream:
            if self._audio_bsf:
                self._audio_bsf_context = av.BitStreamFilterContext(self._audio_bsf, input_astream)
            output_astream = container.add_stream(template=input_astream)
        return (container, output_vstream, output_astream)

    def reset(self, video_dts: int) -> None:
        self._part_start_dts: int = video_dts
        self._segment_start_dts: int = video_dts
        self._segment: Optional[Segment] = None
        self._memory_file: BytesIO = BytesIO()
        self._memory_file_pos: int = 0
        self._av_output, self._output_video_stream, self._output_audio_stream = self.make_new_av(
            memory_file=self._memory_file,
            sequence=self._stream_state.next_sequence(),
            input_vstream=self._input_video_stream,
            input_astream=self._input_audio_stream,
        )
        if self._output_video_stream.name == 'hevc':
            self._output_video_stream.codec_context.codec_tag = 'hvc1'

    def mux_packet(self, packet: av.packet.Packet) -> None:
        if packet.stream == self._input_video_stream:
            if packet.is_keyframe and (packet.dts - self._segment_start_dts) * packet.time_base >= self._stream_settings.min_segment_duration:
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
        self._segment = Segment(
            sequence=self._stream_state.sequence,
            stream_id=self._stream_state.stream_id,
            init=read_init(self._memory_file),
            _stream_outputs=self._stream_state.outputs,
            start_time=self._start_time,
        )
        self._memory_file_pos = self._memory_file.tell()
        self._memory_file.seek(0, SEEK_END)

    def check_flush_part(self, packet: av.packet.Packet) -> None:
        if self._memory_file_pos == self._memory_file.tell():
            return
        if self._segment is None:
            self.create_segment()
            self.flush(packet, last_part=False)
        else:
            self.flush(packet, last_part=False)

    def flush(self, packet: av.packet.Packet, last_part: bool) -> None:
        adjusted_dts: float = min(packet.dts, self._part_start_dts + self._stream_settings.part_target_duration / packet.time_base)
        if last_part:
            self._av_output.close()
            if not self._segment:
                self.create_segment()
        elif not self._part_has_keyframe:
            adjusted_dts = max(adjusted_dts, self._part_start_dts + 0.85 * self._stream_settings.part_target_duration / packet.time_base)
        if not self._stream_settings.ll_hls:
            adjusted_dts = packet.dts  # type: ignore
        assert self._segment is not None
        self._memory_file.seek(self._memory_file_pos)
        segment_duration: float = float((adjusted_dts - self._segment_start_dts) * packet.time_base)
        self._hass.loop.call_soon_threadsafe(
            self._segment.async_add_part,
            Part(
                duration=float((adjusted_dts - self._part_start_dts) * packet.time_base),
                has_keyframe=self._part_has_keyframe,
                data=self._memory_file.read(),
            ),
            segment_duration if last_part else 0,
        )
        if last_part:
            self._memory_file.close()
            self._start_time += datetime.timedelta(seconds=segment_duration)
            self.reset(packet.dts)
        else:
            self._memory_file_pos = self._memory_file.tell()
            self._part_start_dts = adjusted_dts
        self._part_has_keyframe = False

    def close(self) -> None:
        self._av_output.close()
        self._memory_file.close()


class PeekIterator(Iterator[av.packet.Packet]):
    def __init__(self, iterator: Iterator[av.packet.Packet]) -> None:
        self._iterator: Iterator[av.packet.Packet] = iterator
        self._buffer: Deque[av.packet.Packet] = deque()
        self._next = self._iterator.__next__

    def __iter__(self) -> Iterator[av.packet.Packet]:
        return self

    def __next__(self) -> av.packet.Packet:
        return self._next()

    def _pop_buffer(self) -> av.packet.Packet:
        if self._buffer:
            return self._buffer.popleft()
        self._next = self._iterator.__next__
        return self._next()

    def peek(self) -> Generator[av.packet.Packet, None, None]:
        self._next = self._pop_buffer
        yield from self._buffer
        for packet in self._iterator:
            self._buffer.append(packet)
            yield packet


class TimestampValidator:
    def __init__(self, inv_video_time_base: int, inv_audio_time_base: int) -> None:
        self._last_dts: dict[Any, float] = defaultdict(lambda: NEGATIVE_INF)
        self._missing_dts: int = 0
        self._max_dts_gap: float = MAX_TIMESTAMP_GAP * max(inv_video_time_base, inv_audio_time_base)

    def is_valid(self, packet: av.packet.Packet) -> bool:
        if packet.dts is None:
            if self._missing_dts >= MAX_MISSING_DTS:
                raise StreamWorkerError(f'No dts in {MAX_MISSING_DTS + 1} consecutive packets')
            self._missing_dts += 1
            return False
        self._missing_dts = 0
        prev_dts: float = self._last_dts[packet.stream]
        if abs(prev_dts - packet.dts) > self._max_dts_gap and prev_dts != NEGATIVE_INF:
            raise StreamWorkerError(f'Timestamp discontinuity detected: last dts = {prev_dts}, dts = {packet.dts}')
        if packet.dts <= prev_dts:
            return False
        self._last_dts[packet.stream] = packet.dts
        return True


def is_keyframe(packet: av.packet.Packet) -> bool:
    return packet.is_keyframe


def get_audio_bitstream_filter(
    packets: Iterator[av.packet.Packet],
    audio_stream: Optional[av.stream.Stream],
) -> Optional[str]:
    if not audio_stream:
        return None
    for count, packet in enumerate(packets):
        if count >= PACKETS_TO_WAIT_FOR_AUDIO:
            _LOGGER.warning('Audio stream not found')
            break
        if packet.stream == audio_stream:
            if audio_stream.codec.name == 'aac' and packet.size > 2:
                with memoryview(packet) as packet_view:
                    if packet_view[0] == 255 and packet_view[1] & 240 == 240:
                        _LOGGER.debug('ADTS AAC detected. Adding aac_adtstoaac bitstream filter')
                        return 'aac_adtstoasc'
            break
    return None


def try_open_stream(source: str, pyav_options: Mapping[str, Any]) -> InputContainer:
    try:
        return av.open(source, options=pyav_options, timeout=SOURCE_TIMEOUT)
    except av.HTTPBadRequestError as err:
        raise StreamWorkerError(f'Bad Request Error opening stream ({redact_av_error_string(err)})', error_code=StreamClientError.BadRequest) from err
    except av.HTTPUnauthorizedError as err:
        raise StreamWorkerError(f'Unauthorized error opening stream ({redact_av_error_string(err)})', error_code=StreamClientError.Unauthorized) from err
    except av.HTTPForbiddenError as err:
        raise StreamWorkerError(f'Forbidden error opening stream ({redact_av_error_string(err)})', error_code=StreamClientError.Forbidden) from err
    except av.HTTPNotFoundError as err:
        raise StreamWorkerError(f'Not Found error opening stream ({redact_av_error_string(err)})', error_code=StreamClientError.NotFound) from err
    except av.FFmpegError as err:
        raise StreamWorkerError(f'Error opening stream ({redact_av_error_string(err)})') from err


def stream_worker(
    source: str,
    pyav_options: Mapping[str, Any],
    stream_settings: StreamSettings,
    stream_state: StreamState,
    keyframe_converter: KeyFrameConverter,
    quit_event: Event,
) -> None:
    if av.library_versions['libavformat'][0] >= 59 and 'stimeout' in pyav_options:
        pyav_options['timeout'] = pyav_options['stimeout']
        del pyav_options['stimeout']
    container: InputContainer = try_open_stream(source, pyav_options)
    try:
        video_stream: av.stream.Stream = container.streams.video[0]
    except (KeyError, IndexError) as ex:
        raise StreamWorkerError('Stream has no video') from ex
    keyframe_converter.create_codec_context(codec_context=video_stream.codec_context)
    try:
        audio_stream: Optional[av.stream.Stream] = container.streams.audio[0]
    except (KeyError, IndexError):
        audio_stream = None
    if audio_stream and audio_stream.name not in AUDIO_CODECS:
        audio_stream = None
    if audio_stream and audio_stream.profile is None:
        audio_stream = None
    if container.format.name == 'hls':
        for field in fields(StreamSettings):
            setattr(stream_settings, field.name, getattr(STREAM_SETTINGS_NON_LL_HLS, field.name))
    stream_state.diagnostics.set_value('container_format', container.format.name)
    stream_state.diagnostics.set_value('video_codec', video_stream.name)
    if audio_stream:
        stream_state.diagnostics.set_value('audio_codec', audio_stream.name)
    dts_validator: TimestampValidator = TimestampValidator(
        int(1 / video_stream.time_base), int(1 / audio_stream.time_base) if audio_stream else 1
    )
    container_packets: PeekIterator = PeekIterator(filter(dts_validator.is_valid, container.demux((video_stream, audio_stream))))
    
    def is_video(packet: av.packet.Packet) -> bool:
        return packet.stream.type == 'video'
    
    try:
        audio_bsf: Optional[str] = get_audio_bitstream_filter(container_packets.peek(), audio_stream)
        first_keyframe: av.packet.Packet = next(filter(lambda pkt: is_keyframe(pkt) and is_video(pkt), container_packets))
        next_video_packet: av.packet.Packet = next(filter(is_video, container_packets.peek()))
        start_dts: int = next_video_packet.dts - (next_video_packet.duration or 1)  # type: ignore
        first_keyframe.dts = start_dts  # type: ignore
        first_keyframe.pts = start_dts  # type: ignore
    except StreamWorkerError:
        container.close()
        raise
    except StopIteration as ex:
        container.close()
        raise StreamEndedError('Stream ended; no additional packets') from ex
    except av.FFmpegError as ex:
        container.close()
        raise StreamWorkerError(f'Error demuxing stream while finding first packet ({redact_av_error_string(ex)})') from ex
    muxer: StreamMuxer = StreamMuxer(stream_state.hass, video_stream, audio_stream, audio_bsf, stream_state, stream_settings)
    muxer.reset(start_dts)
    muxer.mux_packet(first_keyframe)
    with contextlib.closing(container), contextlib.closing(muxer):
        while not quit_event.is_set():
            try:
                packet: av.packet.Packet = next(container_packets)
            except StreamWorkerError:
                raise
            except StopIteration as ex:
                raise StreamEndedError('Stream ended; no additional packets') from ex
            except av.FFmpegError as ex:
                raise StreamWorkerError(f'Error demuxing stream ({redact_av_error_string(ex)})') from ex
            muxer.mux_packet(packet)
            if packet.is_keyframe and is_video(packet):
                keyframe_converter.stash_keyframe_packet(packet)