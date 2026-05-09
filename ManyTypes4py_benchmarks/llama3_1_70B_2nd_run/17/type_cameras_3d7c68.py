@TYPES.register('Camera')
class Camera(HomeDoorbellAccessory, PyhapCamera):
    """Generate a Camera accessory."""

    def __init__(self, hass: HomeAssistant, driver: HomeDriver, name: str, entity_id: str, aid: int, config: dict[str, Any]):
        """Initialize a Camera accessory object."""
        self._ffmpeg = get_ffmpeg_manager(hass)
        for config_key, conf in CONFIG_DEFAULTS.items():
            if config_key not in config:
                config[config_key] = conf
        max_fps = config[CONF_MAX_FPS]
        max_width = config[CONF_MAX_WIDTH]
        max_height = config[CONF_MAX_HEIGHT]
        resolutions = [(w, h, fps) for w, h, fps in SLOW_RESOLUTIONS if w <= max_width and h <= max_height and (fps < max_fps)] + [(w, h, max_fps) for w, h in RESOLUTIONS if w <= max_width and h <= max_height]
        video_options = {'codec': {'profiles': [VIDEO_CODEC_PARAM_PROFILE_ID_TYPES['BASELINE'], VIDEO_CODEC_PARAM_PROFILE_ID_TYPES['MAIN'], VIDEO_CODEC_PARAM_PROFILE_ID_TYPES['HIGH']], 'levels': [VIDEO_CODEC_PARAM_LEVEL_TYPES['TYPE3_1'], VIDEO_CODEC_PARAM_LEVEL_TYPES['TYPE3_2'], VIDEO_CODEC_PARAM_LEVEL_TYPES['TYPE4_0']]}, 'resolutions': resolutions}
        audio_options = {'codecs': [{'type': 'OPUS', 'samplerate': 24}, {'type': 'OPUS', 'samplerate': 16}]}
        stream_address = config.get(CONF_STREAM_ADDRESS, driver.state.address)
        options = {'video': video_options, 'audio': audio_options, 'address': stream_address, 'srtp': True, 'stream_count': config[CONF_STREAM_COUNT]}
        super().__init__(hass, driver, name, entity_id, aid, config, category=CATEGORY_CAMERA, options=options)
        self._char_motion_detected = None
        self.linked_motion_sensor = self.config.get(CONF_LINKED_MOTION_SENSOR)
        self.motion_is_event = False
        if (linked_motion_sensor := self.linked_motion_sensor):
            self.motion_is_event = linked_motion_sensor.startswith('event.')
            if (state := self.hass.states.get(linked_motion_sensor)):
                serv_motion = self.add_preload_service(SERV_MOTION_SENSOR)
                self._char_motion_detected = serv_motion.configure_char(CHAR_MOTION_DETECTED, value=False)
                self._async_update_motion_state(None, state)

    @pyhap_callback
    @callback
    def run(self) -> None:
        """Handle accessory driver started event.

        Run inside the Home Assistant event loop.
        """
        if self._char_motion_detected:
            assert self.linked_motion_sensor
            self._subscriptions.append(async_track_state_change_event(self.hass, self.linked_motion_sensor, self._async_update_motion_state_event, job_type=HassJobType.Callback))
        super().run()

    @callback
    def _async_update_motion_state_event(self, event: EventStateChangedData) -> None:
        """Handle state change event listener callback."""
        if not state_changed_event_is_same_state(event) and (new_state := event.data['new_state']):
            self._async_update_motion_state(event.data['old_state'], new_state)

    @callback
    def _async_update_motion_state(self, old_state: State | None, new_state: State) -> None:
        """Handle link motion sensor state change to update HomeKit value."""
        state = new_state.state
        char = self._char_motion_detected
        assert char is not None
        if self.motion_is_event:
            if old_state is None or old_state.state == STATE_UNAVAILABLE or state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                return
            _LOGGER.debug('%s: Set linked motion %s sensor to True/False', self.entity_id, self.linked_motion_sensor)
            char.set_value(True)
            char.set_value(False)
            return
        detected = state == STATE_ON
        if char.value == detected:
            return
        char.set_value(detected)
        _LOGGER.debug('%s: Set linked motion %s sensor to %d', self.entity_id, self.linked_motion_sensor, detected)

    @callback
    def async_update_state(self, new_state: State) -> None:
        """Handle state change to update HomeKit value."""

    async def _async_get_stream_source(self) -> str | None:
        """Find the camera stream source url."""
        stream_source = self.config.get(CONF_STREAM_SOURCE)
        if stream_source:
            return stream_source
        try:
            stream_source = await camera.async_get_stream_source(self.hass, self.entity_id)
        except Exception:
            _LOGGER.exception('Failed to get stream source - this could be a transient error or your camera might not be compatible with HomeKit yet')
        return stream_source

    async def start_stream(self, session_info: dict[str, Any], stream_config: dict[str, Any]) -> bool:
        """Start a new stream with the given configuration."""
        _LOGGER.debug('[%s] Starting stream with the following parameters: %s', session_info['id'], stream_config)
        if not (input_source := (await self._async_get_stream_source())):
            _LOGGER.error('Camera has no stream source')
            return False
        if '-i ' not in input_source:
            input_source = '-i ' + input_source
        video_profile = ''
        if self.config[CONF_VIDEO_CODEC] != 'copy':
            video_profile = '-profile:v ' + self.config[CONF_VIDEO_PROFILE_NAMES][int.from_bytes(stream_config['v_profile_id'], byteorder='big')] + ' '
        audio_application = ''
        if self.config[CONF_AUDIO_CODEC] == 'libopus':
            audio_application = '-application lowdelay '
        output_vars = stream_config.copy()
        output_vars.update({'v_profile': video_profile, 'v_bufsize': stream_config['v_max_bitrate'] * 4, 'v_map': self.config[CONF_VIDEO_MAP], 'v_pkt_size': self.config[CONF_VIDEO_PACKET_SIZE], 'v_codec': self.config[CONF_VIDEO_CODEC], 'a_bufsize': stream_config['a_max_bitrate'] * 4, 'a_map': self.config[CONF_AUDIO_MAP], 'a_pkt_size': self.config[CONF_AUDIO_PACKET_SIZE], 'a_encoder': self.config[CONF_AUDIO_CODEC], 'a_application': audio_application})
        output = VIDEO_OUTPUT.format(**output_vars)
        if self.config[CONF_SUPPORT_AUDIO]:
            output = output + ' ' + AUDIO_OUTPUT.format(**output_vars)
        _LOGGER.debug('FFmpeg output settings: %s', output)
        stream = HAFFmpeg(self._ffmpeg.binary)
        opened = await stream.open(cmd=[], input_source=input_source, output=output, extra_cmd='-hide_banner -nostats', stderr_pipe=True, stdout_pipe=False)
        if not opened:
            _LOGGER.error('Failed to open ffmpeg stream')
            return False
        _LOGGER.debug('[%s] Started stream process - PID %d', session_info['id'], stream.process.pid)
        session_info['stream'] = stream
        session_info[FFMPEG_PID] = stream.process.pid
        stderr_reader = await stream.get_reader(source=FFMPEG_STDERR)

        async def watch_session(_) -> None:
            await self._async_ffmpeg_watch(session_info['id'])
        session_info[FFMPEG_LOGGER] = create_eager_task(self._async_log_stderr_stream(stderr_reader))
        session_info[FFMPEG_WATCHER] = async_track_time_interval(self.hass, watch_session, FFMPEG_WATCH_INTERVAL)
        return await self._async_ffmpeg_watch(session_info['id'])

    async def _async_log_stderr_stream(self, stderr_reader: asyncio.StreamReader) -> None:
        """Log output from ffmpeg."""
        _LOGGER.debug('%s: ffmpeg: started', self.display_name)
        while True:
            line = await stderr_reader.readline()
            if line == b'':
                return
            _LOGGER.debug('%s: ffmpeg: %s', self.display_name, line.rstrip())

    async def _async_ffmpeg_watch(self, session_id: str) -> bool:
        """Check to make sure ffmpeg is still running and cleanup if not."""
        ffmpeg_pid = self.sessions[session_id][FFMPEG_PID]
        if pid_is_alive(ffmpeg_pid):
            return True
        _LOGGER.warning('Streaming process ended unexpectedly - PID %d', ffmpeg_pid)
        self._async_stop_ffmpeg_watch(session_id)
        self.set_streaming_available(self.sessions[session_id]['stream_idx'])
        return False

    @callback
    def _async_stop_ffmpeg_watch(self, session_id: str) -> None:
        """Cleanup a streaming session after stopping."""
        if FFMPEG_WATCHER not in self.sessions[session_id]:
            return
        self.sessions[session_id].pop(FFMPEG_WATCHER)()
        self.sessions[session_id].pop(FFMPEG_LOGGER).cancel()

    @callback
    def async_stop(self) -> None:
        """Stop any streams when the accessory is stopped."""
        for session_info in self.sessions.values():
            self.hass.async_create_background_task(self.stop_stream(session_info), 'homekit.camera-stop-stream')
        super().async_stop()

    async def stop_stream(self, session_info: dict[str, Any]) -> None:
        """Stop the stream for the given ``session_id``."""
        session_id = session_info['id']
        if not (stream := session_info.get('stream')):
            _LOGGER.debug('No stream for session ID %s', session_id)
            return
        self._async_stop_ffmpeg_watch(session_id)
        if not pid_is_alive(stream.process.pid):
            _LOGGER.warning('[%s] Stream already stopped', session_id)
            return
        for shutdown_method in ('close', 'kill'):
            _LOGGER.debug('[%s] %s stream', session_id, shutdown_method)
            try:
                await getattr(stream, shutdown_method)()
            except Exception:
                _LOGGER.exception('[%s] Failed to %s stream', session_id, shutdown_method)
            else:
                return

    async def reconfigure_stream(self, session_info: dict[str, Any], stream_config: dict[str, Any]) -> bool:
        """Reconfigure the stream so that it uses the given ``stream_config``."""
        return True

    async def async_get_snapshot(self, image_size: dict[str, int]) -> bytes:
        """Return a jpeg of a snapshot from the camera."""
        image = await camera.async_get_image(self.hass, self.entity_id, width=image_size['image-width'], height=image_size['image-height'])
        return image.content
