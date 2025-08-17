"""Assist satellite entity for Wyoming integration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import io
import logging
from typing import Any, Final, Optional
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.pipeline import PipelineStage, RunPipeline
from wyoming.satellite import PauseSatellite, RunSatellite
from wyoming.snd import Played
from wyoming.timer import TimerCancelled, TimerFinished, TimerStarted, TimerUpdated
from wyoming.tts import Synthesize, SynthesizeVoice
from wyoming.vad import VoiceStarted, VoiceStopped
from wyoming.wake import Detect, Detection

from homeassistant.components import assist_pipeline, ffmpeg, intent, tts
from homeassistant.components.assist_pipeline import PipelineEvent
from homeassistant.components.assist_satellite import (
    AssistSatelliteAnnouncement,
    AssistSatelliteConfiguration,
    AssistSatelliteEntity,
    AssistSatelliteEntityDescription,
    AssistSatelliteEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN, SAMPLE_CHANNELS, SAMPLE_WIDTH
from .data import WyomingService
from .devices import SatelliteDevice
from .entity import WyomingSatelliteEntity
from .models import DomainDataItem

_LOGGER: Final = logging.getLogger(__name__)

_SAMPLES_PER_CHUNK: Final = 1024
_RECONNECT_SECONDS: Final = 10
_RESTART_SECONDS: Final = 3
_PING_TIMEOUT: Final = 5
_PING_SEND_DELAY: Final = 2
_PIPELINE_FINISH_TIMEOUT: Final = 1
_TTS_SAMPLE_RATE: Final = 22050
_ANNOUNCE_CHUNK_BYTES: Final = 2048  # 1024 samples

# Wyoming stage -> Assist stage
_STAGES: dict[PipelineStage, assist_pipeline.PipelineStage] = {
    PipelineStage.WAKE: assist_pipeline.PipelineStage.WAKE_WORD,
    PipelineStage.ASR: assist_pipeline.PipelineStage.STT,
    PipelineStage.HANDLE: assist_pipeline.PipelineStage.INTENT,
    PipelineStage.TTS: assist_pipeline.PipelineStage.TTS,
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Wyoming Assist satellite entity."""
    domain_data: DomainDataItem = hass.data[DOMAIN][config_entry.entry_id]
    assert domain_data.device is not None

    async_add_entities(
        [
            WyomingAssistSatellite(
                hass, domain_data.service, domain_data.device, config_entry
            )
        ]
    )


class WyomingAssistSatellite(WyomingSatelliteEntity, AssistSatelliteEntity):
    """Assist satellite for Wyoming devices."""

    entity_description: AssistSatelliteEntityDescription = AssistSatelliteEntityDescription(key="assist_satellite")
    _attr_translation_key: str = "assist_satellite"
    _attr_name: Optional[str] = None
    _attr_supported_features: AssistSatelliteEntityFeature = AssistSatelliteEntityFeature.ANNOUNCE

    def __init__(
        self,
        hass: HomeAssistant,
        service: WyomingService,
        device: SatelliteDevice,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize an Assist satellite."""
        WyomingSatelliteEntity.__init__(self, device)
        AssistSatelliteEntity.__init__(self)

        self.service: WyomingService = service
        self.device: SatelliteDevice = device
        self.config_entry: ConfigEntry = config_entry

        self.is_running: bool = True

        self._client: Optional[AsyncTcpClient] = None
        self._chunk_converter: AudioChunkConverter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._is_pipeline_running: bool = False
        self._pipeline_ended_event: asyncio.Event = asyncio.Event()
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._pipeline_id: Optional[str] = None
        self._muted_changed_event: asyncio.Event = asyncio.Event()

        self._conversation_id: Optional[str] = None
        self._conversation_id_time: Optional[float] = None

        self.device.set_is_muted_listener(self._muted_changed)
        self.device.set_pipeline_listener(self._pipeline_changed)
        self.device.set_audio_settings_listener(self._audio_settings_changed)

        # For announcements
        self._ffmpeg_manager: Optional[ffmpeg.FFmpegManager] = None
        self._played_event_received: Optional[asyncio.Event] = None

    @property
    def pipeline_entity_id(self) -> Optional[str]:
        """Return the entity ID of the pipeline to use for the next conversation."""
        return self.device.get_pipeline_entity_id(self.hass)

    @property
    def vad_sensitivity_entity_id(self) -> Optional[str]:
        """Return the entity ID of the VAD sensitivity to use for the next conversation."""
        return self.device.get_vad_sensitivity_entity_id(self.hass)

    @property
    def tts_options(self) -> Optional[dict[str, Any]]:
        """Options passed for text-to-speech."""
        return {
            tts.ATTR_PREFERRED_FORMAT: "wav",
            tts.ATTR_PREFERRED_SAMPLE_RATE: _TTS_SAMPLE_RATE,
            tts.ATTR_PREFERRED_SAMPLE_CHANNELS: SAMPLE_CHANNELS,
            tts.ATTR_PREFERRED_SAMPLE_BYTES: SAMPLE_WIDTH,
        }

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        self.start_satellite()

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        await super().async_will_remove_from_hass()
        self.stop_satellite()

    @callback
    def async_get_configuration(self) -> AssistSatelliteConfiguration:
        """Get the current satellite configuration."""
        raise NotImplementedError

    async def async_set_configuration(
        self, config: AssistSatelliteConfiguration
    ) -> None:
        """Set the current satellite configuration."""
        raise NotImplementedError

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Set state based on pipeline stage."""
        assert self._client is not None

        if event.type == assist_pipeline.PipelineEventType.RUN_END:
            # Pipeline run is complete
            self._is_pipeline_running = False
            self._pipeline_ended_event.set()
            self.device.set_is_active(False)
        elif event.type == assist_pipeline.PipelineEventType.WAKE_WORD_START:
            self.hass.add_job(self._client.write_event(Detect().event()))
        elif event.type == assist_pipeline.PipelineEventType.WAKE_WORD_END:
            # Wake word detection
            # Inform client of wake word detection
            if event.data and (wake_word_output := event.data.get("wake_word_output")):
                detection = Detection(
                    name=wake_word_output["wake_word_id"],
                    timestamp=wake_word_output.get("timestamp"),
                )
                self.hass.add_job(self._client.write_event(detection.event()))
        elif event.type == assist_pipeline.PipelineEventType.STT_START:
            # Speech-to-text
            self.device.set_is_active(True)

            if event.data:
                self.hass.add_job(
                    self._client.write_event(
                        Transcribe(language=event.data["metadata"]["language"]).event()
                    )
                )
        elif event.type == assist_pipeline.PipelineEventType.STT_VAD_START:
            # User started speaking
            if event.data:
                self.hass.add_job(
                    self._client.write_event(
                        VoiceStarted(timestamp=event.data["timestamp"]).event()
                    )
                )
        elif event.type == assist_pipeline.PipelineEventType.STT_VAD_END:
            # User stopped speaking
            if event.data:
                self.hass.add_job(
                    self._client.write_event(
                        VoiceStopped(timestamp=event.data["timestamp"]).event()
                    )
                )
        elif event.type == assist_pipeline.PipelineEventType.STT_END:
            # Speech-to-text transcript
            if event.data:
                stt_text: str = event.data["stt_output"]["text"]
                self.hass.add_job(
                    self._client.write_event(Transcript(text=stt_text).event())
                )
        elif event.type == assist_pipeline.PipelineEventType.TTS_START:
            # Text-to-speech text
            if event.data:
                self.hass.add_job(
                    self._client.write_event(
                        Synthesize(
                            text=event.data["tts_input"],
                            voice=SynthesizeVoice(
                                name=event.data.get("voice"),
                                language=event.data.get("language"),
                            ),
                        ).event()
                    )
                )
        elif event.type == assist_pipeline.PipelineEventType.TTS_END:
            # TTS stream
            if event.data and (tts_output := event.data["tts_output"]):
                media_id: str = tts_output["media_id"]
                self.hass.add_job(self._stream_tts(media_id))
        elif event.type == assist_pipeline.PipelineEventType.ERROR:
            # Pipeline error
            if event.data:
                self.hass.add_job(
                    self._client.write_event(
                        Error(
                            text=event.data["message"], code=event.data["code"]
                        ).event()
                    )
                )

    async def async_announce(self, announcement: AssistSatelliteAnnouncement) -> None:
        """Announce media on the satellite.

        Should block until the announcement is done playing.
        """
        assert self._client is not None

        if self._ffmpeg_manager is None:
            self._ffmpeg_manager = ffmpeg.get_ffmpeg_manager(self.hass)

        if self._played_event_received is None:
            self._played_event_received = asyncio.Event()

        self._played_event_received.clear()
        await self._client.write_event(
            AudioStart(
                rate=_TTS_SAMPLE_RATE,
                width=SAMPLE_WIDTH,
                channels=SAMPLE_CHANNELS,
                timestamp=0,
            ).event()
        )

        timestamp: int = 0
        try:
            proc = await asyncio.create_subprocess_exec(
                self._ffmpeg_manager.binary,
                "-i",
                announcement.media_id,
                "-f",
                "s16le",
                "-ac",
                str(SAMPLE_CHANNELS),
                "-ar",
                str(_TTS_SAMPLE_RATE),
                "-nostats",
                "pipe:",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                close_fds=False,  # use posix_spawn in CPython < 3.13
            )
            assert proc.stdout is not None
            while True:
                chunk_bytes: bytes = await proc.stdout.read(_ANNOUNCE_CHUNK_BYTES)
                if not chunk_bytes:
                    break

                chunk = AudioChunk(
                    rate=_TTS_SAMPLE_RATE,
                    width=SAMPLE_WIDTH,
                    channels=SAMPLE_CHANNELS,
                    audio=chunk_bytes,
                    timestamp=timestamp,
                )
                await self._client.write_event(chunk.event())

                timestamp += chunk.milliseconds
        finally:
            await self._client.write_event(AudioStop().event())
            if timestamp > 0:
                audio_seconds: float = timestamp / 1000
                try:
                    async with asyncio.timeout(audio_seconds + 0.5):
                        await self._played_event_received.wait()
                except TimeoutError:
                    _LOGGER.debug("Did not receive played event for announcement")

    def start_satellite(self) -> None:
        """Start satellite task."""
        self.is_running = True

        self.config_entry.async_create_background_task(
            self.hass, self.run(), "wyoming satellite run"
        )

    def stop_satellite(self) -> None:
        """Signal satellite task to stop running."""
        self._audio_queue.put_nowait(None)
        self._send_pause()
        self.is_running = False
        self._muted_changed_event.set()

    async def run(self) -> None:
        """Run and maintain a connection to satellite."""
        _LOGGER.debug("Running satellite task")

        unregister_timer_handler = intent.async_register_timer_handler(
            self.hass, self.device.device_id, self._handle_timer
        )

        try:
            while self.is_running:
                try:
                    while self.device.is_muted:
                        _LOGGER.debug("Satellite is muted")
                        await self.on_muted()
                        if not self.is_running:
                            return

                    await self._connect_and_loop()
                except asyncio.CancelledError:
                    raise
                except Exception as err:  # noqa: BLE001
                    _LOGGER.debug("%s: %s", err.__class__.__name__, str(err))
                    self._audio_queue.put_nowait(None)
                    self.device.set_is_active(False)
                    await self.on_restart()
        finally:
            unregister_timer_handler()
            self.device.set_is_active(False)
            await self.on_stopped()

    async def on_restart(self) -> None:
        """Block until pipeline loop will be restarted."""
        _LOGGER.warning(
            "Satellite has been disconnected. Reconnecting in %s second(s)",
            _RECONNECT_SECONDS,
        )
        await asyncio.sleep(_RESTART_SECONDS)

    async def on_reconnect(self) -> None:
        """Block until a reconnection attempt should be made."""
        _LOGGER.debug(
            "Failed to connect to satellite. Reconnecting in %s second(s)",
            _RECONNECT_SECONDS,
        )
        await asyncio.sleep(_RECONNECT_SECONDS)

    async def on_muted(self) -> None:
        """Block until device may be unmuted again."""
        await self._muted_changed_event.wait()

    async def on_stopped(self) -> None:
        """Run when run() has fully stopped."""
        _LOGGER.debug("Satellite task stopped")

    def _send_pause(self) -> None:
        """Send a pause message to satellite."""
        if self._client is not None:
            self.config_entry.async_create_background_task(
                self.hass,
                self._client.write_event(PauseSatellite().event()),
                "pause satellite",
            )

    def _muted_changed(self) -> None:
        """Run when device muted status changes."""
        if self.device.is_muted:
            self._audio_queue.put_nowait(None)
            self._send_pause()

        self._muted_changed_event.set()
        self._muted_changed_event.clear()

    def _pipeline_changed(self) -> None:
        """Run when device pipeline changes."""
        self._audio_queue.put_nowait(None)

    def _audio_settings_changed(self) -> None:
        """Run when device audio settings."""
        self._audio_queue.put_nowait(None)

    async def _connect_and_loop(self) -> None:
        """Connect to satellite and run pipelines until an error occurs."""
        while self.is_running and (not self.device.is_muted):
            try:
                await self._connect()
                break
            except ConnectionError:
                self._client = None
                await self.on_reconnect()

        if self._client is None:
            return

        _LOGGER.debug("Connected to satellite")

        if (not self.is_running) or self.device.is_muted:
            return

        await self._client.write_event(RunSatellite().event())

        while self.is_running and (not self.device.is_muted):
            await self._run_pipeline_loop()

    async def _run_pipeline_loop(self) -> None:
        """Run a pipeline one or more times."""
        assert self._client is not None
        client_info: Optional[Info] = None
        wake_word_phrase: Optional[str] = None
        run_pipeline: Optional[RunPipeline] = None
        send_ping: bool = True

        pipeline_ended_task: asyncio.Task[None] = self.config_entry.async_create_background_task(
            self.hass, self._pipeline_ended_event.wait(), "satellite pipeline ended"
        )
        client_event_task: asyncio.Task[Optional[Event]] = self.config_entry.async_create_background_task(
            self.hass, self._client.read_event(), "satellite event read"
        )
        pending: set[asyncio.Task] = {pipeline_ended_task, client_event_task}

        await self._client.write_event(Describe().event())

        while self.is_running and (not self.device.is_muted):
            if send_ping:
                send_ping = False
                self.config_entry.async_create_background_task(
                    self.hass, self._send_delayed_ping(), "ping satellite"
                )

            async with asyncio.timeout(_PING_TIMEOUT):
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                if pipeline_ended_task in done:
                    _LOGGER.debug("Pipeline finished")
                    self._pipeline_ended_event.clear()
                    pipeline_ended_task = self.config_entry.async_create_background_task(
                        self.hass,
                        self._pipeline_ended_event.wait(),
                        "satellite pipeline ended",
                    )
                    pending.add(pipeline_ended_task)

                    wake_word_phrase = None

                    if (run_pipeline is not None) and run_pipeline.restart_on_end:
                        self._run_pipeline_once(run_pipeline)
                        continue

                if client_event_task not in done:
                    continue

                client_event: Optional[Event] = client_event_task.result()
                if client_event is None:
                    raise ConnectionResetError("Satellite disconnected")

                if Pong.is_type(client_event.type):
                    send_ping = True
                elif Ping.is_type(client_event.type):
                    ping = Ping.from_event(client_event)
                    await self._client.write_event(Pong(text=ping.text).event())
                elif RunPipeline.is_type(client_event.type):
                    run_pipeline = RunPipeline.from_event(client_event)
                    self._run_pipeline_once(run_pipeline, wake_word_phrase)
                elif (
                    AudioChunk.is_type(client_event.type) and self._is_pipeline_running
                ):
                    chunk = AudioChunk.from_event(client_event)
                    chunk = self._chunk_converter.convert(chunk)
                    self._audio_queue.put_nowait(chunk.audio)
                elif AudioStop.is_type(client_event.type) and self._is_pipeline_running:
                    _LOGGER.debug("Client requested pipeline to stop")
                    self._audio_queue.put_nowait(None)
                elif Info.is_type(client_event.type):
                    client_info = Info.from_event(client_event)
                    _LOGGER.debug("Updated client info: %s", client_info)
                elif Detection.is_type(client_event.type):
                    detection = Detection.from_event(client_event)
                    wake_word_phrase = detection.name

                    if (client_info is not None) and (client_info.wake is not None):
                        found_phrase: bool = False
                        for wake_service in client_info.wake:
                            for wake_model in wake_service.models:
                                if wake_model.name == detection.name:
                                    wake_word_phrase = wake_model.phrase or wake_model.name
                                    found_phrase = True
                                    break
                            if found_phrase:
                                break
                    _LOGGER.debug("Client detected wake word: %s", wake_word_phrase)
                elif Played.is_type(client_event.type):
                    self.tts_response_finished()
                    if self._played_event_received is not None:
                        self._played_event_received.set()
                else:
                    _LOGGER.debug("Unexpected event from satellite: %s", client_event)

                client_event_task = self.config_entry.async_create_background_task(
                    self.hass, self._client.read_event(), "satellite event read"
                )
                pending.add(client_event_task)

    def _run_pipeline_once(
        self, run_pipeline: RunPipeline, wake_word_phrase: Optional[str] = None
    ) -> None:
        """Run a pipeline once."""
        _LOGGER.debug("Received run information: %s", run_pipeline)

        start_stage: Optional[assist_pipeline.PipelineStage] = _STAGES.get(run_pipeline.start_stage)
        end_stage: Optional[assist_pipeline.PipelineStage] = _STAGES.get(run_pipeline.end_stage)

        if start_stage is None:
            raise ValueError(f"Invalid start stage: {start_stage}")

        if end_stage is None:
            raise ValueError(f"Invalid end stage: {end_stage}")

        self._audio_queue = asyncio.Queue()
        self._is_pipeline_running = True
        self._pipeline_ended_event.clear()
        self.config_entry.async_create_background_task(
            self.hass,
            self.async_accept_pipeline_from_satellite(
                audio_stream=self._stt_stream(),
                start_stage=start_stage,
                end_stage=end_stage,
                wake_word_phrase=wake_word_phrase,
            ),
            "wyoming satellite pipeline",
        )

    async def _send_delayed_ping(self) -> None:
        """Send ping to satellite after a delay."""
        assert self._client is not None
        try:
            await asyncio.sleep(_PING_SEND_DELAY)
            await self._client.write_event(Ping().event())
        except ConnectionError:
            pass

    async def _connect(self) -> None:
        """Connect to satellite over TCP."""
        await self._disconnect()
        _LOGGER.debug(
            "Connecting to satellite at %s:%s", self.service.host, self.service.port
        )
        self._client = AsyncTcpClient(self.service.host, self.service.port)
        await self._client.connect()

    async def _disconnect(self) -> None:
        """Disconnect if satellite is currently connected."""
        if self._client is None:
            return
        _LOGGER.debug("Disconnecting from satellite")
        await self._client.disconnect()
        self._client = None

    async def _stream_tts(self, media_id: str) -> None:
        """Stream TTS WAV audio to satellite in chunks."""
        assert self._client is not None

        extension, data = await tts.async_get_media_source_audio(self.hass, media_id)
        if extension != "wav":
            raise ValueError(f"Cannot stream audio format to satellite: {extension}")

        with io.BytesIO(data) as wav_io, wave.open(wav_io, "rb") as wav_file:
            sample_rate: int = wav_file.getframerate()
            sample_width: int = wav_file.getsampwidth()
            sample_channels: int = wav_file.getnchannels()
            _LOGGER.debug("Streaming %s TTS sample(s)", wav_file.getnframes())

            timestamp: int = 0
            await self._client.write_event(
                AudioStart(
                    rate=sample_rate,
                    width=sample_width,
                    channels=sample_channels,
                    timestamp=timestamp,
                ).event()
            )

            while audio_bytes := wav_file.readframes(_SAMPLES_PER_CHUNK):
                chunk = AudioChunk(
                    rate=sample_rate,
                    width=sample_width,
                    channels=sample_channels,
                    audio=audio_bytes,
                    timestamp=timestamp,
                )
                await self._client.write_event(chunk.event())
                timestamp += chunk.seconds

            await self._client.write_event(AudioStop(timestamp=timestamp).event())
            _LOGGER.debug("TTS streaming complete")

    async def _stt_stream(self) -> AsyncGenerator[bytes, None]:
        """Yield audio chunks from a queue."""
        is_first_chunk: bool = True
        while (chunk := await self._audio_queue.get()) is not None:
            if is_first_chunk:
                is_first_chunk = False
                _LOGGER.debug("Receiving audio from satellite")
            yield chunk

    @callback
    def _handle_timer(
        self, event_type: intent.TimerEventType, timer: intent.TimerInfo
    ) -> None:
        """Forward timer events to satellite."""
        assert self._client is not None

        _LOGGER.debug("Timer event: type=%s, info=%s", event_type, timer)
        event: Optional[Event] = None
        if event_type == intent.TimerEventType.STARTED:
            event = TimerStarted(
                id=timer.id,
                total_seconds=timer.seconds,
                name=timer.name,
                start_hours=timer.start_hours,
                start_minutes=timer.start_minutes,
                start_seconds=timer.start_seconds,
            ).event()
        elif event_type == intent.TimerEventType.UPDATED:
            event = TimerUpdated(
                id=timer.id,
                is_active=timer.is_active,
                total_seconds=timer.seconds,
            ).event()
        elif event_type == intent.TimerEventType.CANCELLED:
            event = TimerCancelled(id=timer.id).event()
        elif event_type == intent.TimerEventType.FINISHED:
            event = TimerFinished(id=timer.id).event()

        if event is not None:
            self.config_entry.async_create_background_task(
                self.hass, self._client.write_event(event), "wyoming timer event"
            )