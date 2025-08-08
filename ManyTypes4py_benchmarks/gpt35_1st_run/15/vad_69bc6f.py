from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
import logging
from .const import SAMPLE_CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
_LOGGER: logging.Logger = logging.getLogger(__name__)

class VadSensitivity(StrEnum):
    DEFAULT: str = 'default'
    RELAXED: str = 'relaxed'
    AGGRESSIVE: str = 'aggressive'

    @staticmethod
    def to_seconds(sensitivity: VadSensitivity) -> float:
        sensitivity = VadSensitivity(sensitivity)
        if sensitivity == VadSensitivity.RELAXED:
            return 1.25
        if sensitivity == VadSensitivity.AGGRESSIVE:
            return 0.25
        return 0.7

class AudioBuffer:
    def __init__(self, maxlen: int) -> None:
        self._buffer: bytearray = bytearray(maxlen)
        self._length: int = 0

    @property
    def length(self) -> int:
        return self._length

    def clear(self) -> None:
        self._length = 0

    def append(self, data: bytes) -> None:
        data_len: int = len(data)
        if self._length + data_len > len(self._buffer):
            raise ValueError('Length cannot be greater than buffer size')
        self._buffer[self._length:self._length + data_len] = data
        self._length += data_len

    def bytes(self) -> bytes:
        return bytes(self._buffer[:self._length])

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        return self._length > 0

@dataclass
class VoiceCommandSegmenter:
    speech_seconds: float = 0.3
    command_seconds: float = 1.0
    silence_seconds: float = 0.7
    timeout_seconds: float = 15.0
    reset_seconds: float = 1.0
    in_command: bool = False
    timed_out: bool = False
    before_command_speech_threshold: float = 0.2
    in_command_speech_threshold: float = 0.5
    _speech_seconds_left: float = 0.0
    _command_seconds_left: float = 0.0
    _silence_seconds_left: float = 0.0
    _timeout_seconds_left: float = 0.0
    _reset_seconds_left: float = 0.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._speech_seconds_left = self.speech_seconds
        self._command_seconds_left = self.command_seconds - self.speech_seconds
        self._silence_seconds_left = self.silence_seconds
        self._timeout_seconds_left = self.timeout_seconds
        self._reset_seconds_left = self.reset_seconds
        self.in_command = False

    def process(self, chunk_seconds: float, speech_probability: float) -> bool:
        if self.timed_out:
            self.timed_out = False
        self._timeout_seconds_left -= chunk_seconds
        if self._timeout_seconds_left <= 0:
            _LOGGER.debug('VAD end of speech detection timed out after %s seconds', self.timeout_seconds)
            self.reset()
            self.timed_out = True
            return False
        if speech_probability is None:
            speech_probability = 0.0
        if not self.in_command:
            is_speech = speech_probability > self.before_command_speech_threshold
            if is_speech:
                self._reset_seconds_left = self.reset_seconds
                self._speech_seconds_left -= chunk_seconds
                if self._speech_seconds_left <= 0:
                    self.in_command = True
                    self._command_seconds_left = self.command_seconds - self.speech_seconds
                    self._silence_seconds_left = self.silence_seconds
                    _LOGGER.debug('Voice command started')
            else:
                self._reset_seconds_left -= chunk_seconds
                if self._reset_seconds_left <= 0:
                    self._speech_seconds_left = self.speech_seconds
                    self._reset_seconds_left = self.reset_seconds
        else:
            is_speech = speech_probability > self.in_command_speech_threshold
            if not is_speech:
                self._reset_seconds_left = self.reset_seconds
                self._silence_seconds_left -= chunk_seconds
                self._command_seconds_left -= chunk_seconds
                if self._silence_seconds_left <= 0 and self._command_seconds_left <= 0:
                    self.reset()
                    _LOGGER.debug('Voice command finished')
                    return False
            else:
                self._reset_seconds_left -= chunk_seconds
                self._command_seconds_left -= chunk_seconds
                if self._reset_seconds_left <= 0:
                    self._silence_seconds_left = self.silence_seconds
                    self._reset_seconds_left = self.reset_seconds
        return True

    def process_with_vad(self, chunk: bytes, vad_samples_per_chunk: int, vad_is_speech: Callable, leftover_chunk_buffer: AudioBuffer) -> bool:
        if vad_samples_per_chunk is None:
            chunk_seconds = len(chunk) // (SAMPLE_WIDTH * SAMPLE_CHANNELS) / SAMPLE_RATE
            is_speech = vad_is_speech(chunk)
            return self.process(chunk_seconds, is_speech)
        if leftover_chunk_buffer is None:
            raise ValueError('leftover_chunk_buffer is required when vad uses chunking')
        seconds_per_chunk = vad_samples_per_chunk / SAMPLE_RATE
        bytes_per_chunk = vad_samples_per_chunk * (SAMPLE_WIDTH * SAMPLE_CHANNELS)
        for vad_chunk in chunk_samples(chunk, bytes_per_chunk, leftover_chunk_buffer):
            is_speech = vad_is_speech(vad_chunk)
            if not self.process(seconds_per_chunk, is_speech):
                return False
        return True

@dataclass
class VoiceActivityTimeout:
    reset_seconds: float = 0.5
    speech_threshold: float = 0.5
    _silence_seconds_left: float = 0.0
    _reset_seconds_left: float = 0.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._silence_seconds_left = self.silence_seconds
        self._reset_seconds_left = self.reset_seconds

    def process(self, chunk_seconds: float, speech_probability: float) -> bool:
        if speech_probability is None:
            speech_probability = 0.0
        if speech_probability > self.speech_threshold:
            self._reset_seconds_left -= chunk_seconds
            if self._reset_seconds_left <= 0:
                self._silence_seconds_left = self.silence_seconds
        else:
            self._silence_seconds_left -= chunk_seconds
            if self._silence_seconds_left <= 0:
                self.reset()
                return False
            self._reset_seconds_left = min(self.reset_seconds, self._reset_seconds_left + chunk_seconds)
        return True

def chunk_samples(samples: bytes, bytes_per_chunk: int, leftover_chunk_buffer: AudioBuffer) -> None:
    if len(leftover_chunk_buffer) + len(samples) < bytes_per_chunk:
        leftover_chunk_buffer.append(samples)
        return
    next_chunk_idx = 0
    if leftover_chunk_buffer:
        bytes_to_copy = bytes_per_chunk - len(leftover_chunk_buffer)
        leftover_chunk_buffer.append(samples[:bytes_to_copy])
        next_chunk_idx = bytes_to_copy
        yield leftover_chunk_buffer.bytes()
        leftover_chunk_buffer.clear()
    while next_chunk_idx < len(samples) - bytes_per_chunk + 1:
        yield samples[next_chunk_idx:next_chunk_idx + bytes_per_chunk]
        next_chunk_idx += bytes_per_chunk
    if (rest_samples := samples[next_chunk_idx:]):
        leftover_chunk_buffer.append(rest_samples)
