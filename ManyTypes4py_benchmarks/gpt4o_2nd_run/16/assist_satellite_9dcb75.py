"""Assist satellite entity for VoIP integration."""
from __future__ import annotations
import asyncio
from enum import IntFlag
from functools import partial
import io
import logging
from pathlib import Path
import socket
import time
from typing import TYPE_CHECKING, Any, Final, Optional, Union, AsyncGenerator
import wave
from voip_utils import SIP_PORT, RtpDatagramProtocol
from voip_utils.sip import SipDatagramProtocol, SipEndpoint, get_sip_endpoint
from homeassistant.components import tts
from homeassistant.components.assist_pipeline import PipelineEvent, PipelineEventType
from homeassistant.components.assist_satellite import AssistSatelliteAnnouncement, AssistSatelliteConfiguration, AssistSatelliteEntity, AssistSatelliteEntityDescription, AssistSatelliteEntityFeature
from homeassistant.components.network import async_get_source_ip
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import CHANNELS, CONF_SIP_PORT, CONF_SIP_USER, DOMAIN, RATE, RTP_AUDIO_SETTINGS, WIDTH
from .devices import VoIPDevice
from .entity import VoIPEntity
if TYPE_CHECKING:
    from . import DomainData

_LOGGER = logging.getLogger(__name__)
_PIPELINE_TIMEOUT_SEC: Final[int] = 30
_ANNOUNCEMENT_BEFORE_DELAY: Final[float] = 0.5
_ANNOUNCEMENT_AFTER_DELAY: Final[float] = 1.0
_ANNOUNCEMENT_HANGUP_SEC: Final[float] = 0.5
_ANNOUNCEMENT_RING_TIMEOUT: Final[int] = 30

class Tones(IntFlag):
    """Feedback tones for specific events."""
    LISTENING = 1
    PROCESSING = 2
    ERROR = 4

_TONE_FILENAMES: Final[dict[Tones, str]] = {
    Tones.LISTENING: 'tone.pcm',
    Tones.PROCESSING: 'processing.pcm',
    Tones.ERROR: 'error.pcm'
}

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up VoIP Assist satellite entity."""
    domain_data: DomainData = hass.data[DOMAIN]

    @callback
    def async_add_device(device: VoIPDevice) -> None:
        """Add device."""
        async_add_entities([VoipAssistSatellite(hass, device, config_entry)])

    domain_data.devices.async_add_new_device_listener(async_add_device)
    entities = [VoipAssistSatellite(hass, device, config_entry) for device in domain_data.devices]
    async_add_entities(entities)

class VoipAssistSatellite(VoIPEntity, AssistSatelliteEntity, RtpDatagramProtocol):
    """Assist satellite for VoIP devices."""
    entity_description: AssistSatelliteEntityDescription = AssistSatelliteEntityDescription(key='assist_satellite')
    _attr_translation_key: str = 'assist_satellite'
    _attr_name: Optional[str] = None
    _attr_supported_features: AssistSatelliteEntityFeature = AssistSatelliteEntityFeature.ANNOUNCE | AssistSatelliteEntityFeature.START_CONVERSATION

    def __init__(
        self,
        hass: HomeAssistant,
        voip_device: VoIPDevice,
        config_entry: ConfigEntry,
        tones: Tones = Tones.LISTENING | Tones.PROCESSING | Tones.ERROR
    ) -> None:
        """Initialize an Assist satellite."""
        VoIPEntity.__init__(self, voip_device)
        AssistSatelliteEntity.__init__(self)
        RtpDatagramProtocol.__init__(self)
        self.config_entry: ConfigEntry = config_entry
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._audio_chunk_timeout: float = 2.0
        self._run_pipeline_task: Optional[asyncio.Task] = None
        self._pipeline_had_error: bool = False
        self._tts_done: asyncio.Event = asyncio.Event()
        self._tts_extra_timeout: float = 1.0
        self._tone_bytes: dict[Tones, bytes] = {}
        self._tones: Tones = tones
        self._processing_tone_done: asyncio.Event = asyncio.Event()
        self._announcement: Optional[AssistSatelliteAnnouncement] = None
        self._announcement_future: asyncio.Future = asyncio.Future()
        self._announcment_start_time: float = 0.0
        self._check_announcement_ended_task: Optional[asyncio.Task] = None
        self._last_chunk_time: Optional[float] = None
        self._rtp_port: Optional[int] = None
        self._run_pipeline_after_announce: bool = False

    @property
    def pipeline_entity_id(self) -> Optional[str]:
        """Return the entity ID of the pipeline to use for the next conversation."""
        return self.voip_device.get_pipeline_entity_id(self.hass)

    @property
    def vad_sensitivity_entity_id(self) -> Optional[str]:
        """Return the entity ID of the VAD sensitivity to use for the next conversation."""
        return self.voip_device.get_vad_sensitivity_entity_id(self.hass)

    @property
    def tts_options(self) -> dict[str, Union[str, int]]:
        """Options passed for text-to-speech."""
        return {
            tts.ATTR_PREFERRED_FORMAT: 'wav',
            tts.ATTR_PREFERRED_SAMPLE_RATE: 16000,
            tts.ATTR_PREFERRED_SAMPLE_CHANNELS: 1,
            tts.ATTR_PREFERRED_SAMPLE_BYTES: 2
        }

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        self.voip_device.protocol = self

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        await super().async_will_remove_from_hass()
        assert self.voip_device.protocol == self
        self.voip_device.protocol = None

    @callback
    def async_get_configuration(self) -> AssistSatelliteConfiguration:
        """Get the current satellite configuration."""
        raise NotImplementedError

    async def async_set_configuration(self, config: AssistSatelliteConfiguration) -> None:
        """Set the current satellite configuration."""
        raise NotImplementedError

    async def async_announce(self, announcement: AssistSatelliteAnnouncement) -> None:
        """Announce media on the satellite.

        Plays announcement in a loop, blocking until the caller hangs up.
        """
        await self._do_announce(announcement, run_pipeline_after=False)

    async def _do_announce(self, announcement: AssistSatelliteAnnouncement, run_pipeline_after: bool) -> None:
        """Announce media on the satellite.

        Optionally run a voice pipeline after the announcement has finished.
        """
        self._announcement_future = asyncio.Future()
        self._run_pipeline_after_announce = run_pipeline_after
        if self._rtp_port is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind(('', 0))
            _rtp_ip, self._rtp_port = sock.getsockname()
            sock.close()
        source_ip = await async_get_source_ip(self.hass)
        sip_port = self.config_entry.options.get(CONF_SIP_PORT, SIP_PORT)
        sip_user = self.config_entry.options.get(CONF_SIP_USER)
        source_endpoint = get_sip_endpoint(host=source_ip, port=sip_port, username=sip_user)
        try:
            destination_endpoint = SipEndpoint(self.voip_device.voip_id)
        except ValueError:
            destination_endpoint = get_sip_endpoint(host=self.voip_device.voip_id, port=SIP_PORT)
        self._last_chunk_time = None
        self._announcment_start_time = time.monotonic()
        self._announcement = announcement
        sip_protocol: SipDatagramProtocol = self.hass.data[DOMAIN].protocol
        call_info = sip_protocol.outgoing_call(source=source_endpoint, destination=destination_endpoint, rtp_port=self._rtp_port)
        self._check_announcement_ended_task = self.config_entry.async_create_background_task(self.hass, self._check_announcement_ended(), 'voip_announcement_ended')
        try:
            await self._announcement_future
        except TimeoutError:
            sip_protocol.cancel_call(call_info)
            raise

    async def _check_announcement_ended(self) -> None:
        """Continuously checks if an audio chunk was received within a time limit.

        If not, the caller is presumed to have hung up and the announcement is ended.
        """
        while self._announcement is not None:
            current_time = time.monotonic()
            if self._last_chunk_time is None and current_time - self._announcment_start_time > _ANNOUNCEMENT_RING_TIMEOUT:
                self._announcement = None
                self._check_announcement_ended_task = None
                self._announcement_future.set_exception(TimeoutError('User did not pick up in time'))
                _LOGGER.debug('Timed out waiting for the user to pick up the phone')
                break
            if self._last_chunk_time is not None and current_time - self._last_chunk_time > _ANNOUNCEMENT_HANGUP_SEC:
                self._announcement = None
                self._announcement_future.set_result(None)
                self._check_announcement_ended_task = None
                _LOGGER.debug('Announcement ended')
                break
            await asyncio.sleep(_ANNOUNCEMENT_HANGUP_SEC / 2)

    async def async_start_conversation(self, start_announcement: AssistSatelliteAnnouncement) -> None:
        """Start a conversation from the satellite."""
        await self._do_announce(start_announcement, run_pipeline_after=True)

    def on_chunk(self, audio_bytes: bytes) -> None:
        """Handle raw audio chunk."""
        self._last_chunk_time = time.monotonic()
        if self._announcement is None:
            if self._run_pipeline_task is None:
                self._clear_audio_queue()
                self._tts_done.clear()
                self._run_pipeline_task = self.config_entry.async_create_background_task(self.hass, self._run_pipeline(), 'voip_pipeline_run')
            self._audio_queue.put_nowait(audio_bytes)
        elif self._run_pipeline_task is None:
            self._run_pipeline_task = self.config_entry.async_create_background_task(self.hass, self._play_announcement(self._announcement), 'voip_play_announcement')

    async def _run_pipeline(self) -> None:
        """Run a pipeline with STT input and TTS output."""
        _LOGGER.debug('Starting pipeline')
        self.async_set_context(Context(user_id=self.config_entry.data['user']))
        self.voip_device.set_is_active(True)

        async def stt_stream() -> AsyncGenerator[bytes, None]:
            while True:
                async with asyncio.timeout(self._audio_chunk_timeout):
                    chunk = await self._audio_queue.get()
                    if not chunk:
                        break
                    yield chunk

        await self._play_tone(Tones.LISTENING, silence_before=0.2)
        try:
            await self.async_accept_pipeline_from_satellite(audio_stream=stt_stream())
            if self._pipeline_had_error:
                self._pipeline_had_error = False
                await self._play_tone(Tones.ERROR)
            else:
                await self._tts_done.wait()
        except TimeoutError:
            self.disconnect()
        finally:
            await self._audio_queue.put(None)
            self.voip_device.set_is_active(False)
            self._run_pipeline_task = None
            _LOGGER.debug('Pipeline finished')

    async def _play_announcement(self, announcement: AssistSatelliteAnnouncement) -> None:
        """Play an announcement once."""
        _LOGGER.debug('Playing announcement')
        try:
            await asyncio.sleep(_ANNOUNCEMENT_BEFORE_DELAY)
            await self._send_tts(announcement.original_media_id, wait_for_tone=False)
            if not self._run_pipeline_after_announce:
                await asyncio.sleep(_ANNOUNCEMENT_AFTER_DELAY)
        except Exception:
            _LOGGER.exception('Unexpected error while playing announcement')
            raise
        finally:
            self._run_pipeline_task = None
            _LOGGER.debug('Announcement finished')
            if self._run_pipeline_after_announce:
                self._announcement = None
                self._announcement_future.set_result(None)

    def _clear_audio_queue(self) -> None:
        """Ensure audio queue is empty."""
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Set state based on pipeline stage."""
        if event.type == PipelineEventType.STT_END:
            if self._tones & Tones.PROCESSING == Tones.PROCESSING:
                self._processing_tone_done.clear()
                self.config_entry.async_create_background_task(self.hass, self._play_tone(Tones.PROCESSING), 'voip_process_tone')
        elif event.type == PipelineEventType.TTS_END:
            if event.data and (tts_output := event.data['tts_output']):
                media_id = tts_output['media_id']
                self.config_entry.async_create_background_task(self.hass, self._send_tts(media_id), 'voip_pipeline_tts')
            else:
                self._tts_done.set()
        elif event.type == PipelineEventType.ERROR:
            self._pipeline_had_error = True
            _LOGGER.warning(event)

    async def _send_tts(self, media_id: str, wait_for_tone: bool = True) -> None:
        """Send TTS audio to caller via RTP."""
        try:
            if self.transport is None:
                return
            extension, data = await tts.async_get_media_source_audio(self.hass, media_id)
            if extension != 'wav':
                raise ValueError(f'Only WAV audio can be streamed, got {extension}')
            if wait_for_tone and self._tones & Tones.PROCESSING == Tones.PROCESSING:
                _LOGGER.debug('Waiting for processing tone')
                await self._processing_tone_done.wait()
            with io.BytesIO(data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    sample_width = wav_file.getsampwidth()
                    sample_channels = wav_file.getnchannels()
                    if sample_rate != RATE or sample_width != WIDTH or sample_channels != CHANNELS:
                        raise ValueError(f'Expected rate/width/channels as {RATE}/{WIDTH}/{CHANNELS}, got {sample_rate}/{sample_width}/{sample_channels}')
                    audio_bytes = wav_file.readframes(wav_file.getnframes())
            _LOGGER.debug('Sending %s byte(s) of audio', len(audio_bytes))
            tts_samples = len(audio_bytes) / (WIDTH * CHANNELS)
            tts_seconds = tts_samples / RATE
            async with asyncio.timeout(tts_seconds + self._tts_extra_timeout):
                await self._async_send_audio(audio_bytes)
        except TimeoutError:
            _LOGGER.warning('TTS timeout')
            raise
        finally:
            self.tts_response_finished()
            self._tts_done.set()

    async def _async_send_audio(self, audio_bytes: bytes, **kwargs: Any) -> None:
        """Send audio in executor."""
        await self.hass.async_add_executor_job(partial(self.send_audio, audio_bytes, **RTP_AUDIO_SETTINGS, **kwargs))

    async def _play_tone(self, tone: Tones, silence_before: float = 0.0) -> None:
        """Play a tone as feedback to the user if it's enabled."""
        if self._tones & tone != tone:
            return
        if tone not in self._tone_bytes:
            self._tone_bytes[tone] = await self.hass.async_add_executor_job(self._load_pcm, _TONE_FILENAMES[tone])
        await self._async_send_audio(self._tone_bytes[tone], silence_before=silence_before)
        if tone == Tones.PROCESSING:
            self._processing_tone_done.set()

    def _load_pcm(self, file_name: str) -> bytes:
        """Load raw audio (16Khz, 16-bit mono)."""
        return (Path(__file__).parent / file_name).read_bytes()
