class VoipAssistSatellite(VoIPEntity, AssistSatelliteEntity, RtpDatagramProtocol):
    """Assist satellite for VoIP devices."""

    entity_description: AssistSatelliteEntityDescription
    _attr_translation_key: str
    _attr_name: str | None
    _attr_supported_features: AssistSatelliteEntityFeature

    def __init__(self, hass: HomeAssistant, voip_device: VoIPDevice, config_entry: ConfigEntry, tones: Tones = Tones.LISTENING | Tones.PROCESSING | Tones.ERROR):
        """Initialize an Assist satellite."""
        # ... (rest of the method remains the same)

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
        """Announce media on the satellite."""
        await self._do_announce(announcement, run_pipeline_after=False)

    async def _do_announce(self, announcement: AssistSatelliteAnnouncement, run_pipeline_after: bool) -> None:
        # ... (rest of the method remains the same)

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
        # ... (rest of the method remains the same)

    async def _play_announcement(self, announcement: AssistSatelliteAnnouncement) -> None:
        """Play an announcement once."""
        # ... (rest of the method remains the same)

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
        # ... (rest of the method remains the same)

    async def _async_send_audio(self, audio_bytes: bytes, **kwargs) -> None:
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
