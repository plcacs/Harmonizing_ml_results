from __future__ import annotations
import asyncio
from datetime import timedelta
import logging
from typing import Any, Optional, Dict, List, Tuple
from haffmpeg.core import FFMPEG_STDERR, HAFFmpeg
from pyhap.camera import VIDEO_CODEC_PARAM_LEVEL_TYPES, VIDEO_CODEC_PARAM_PROFILE_ID_TYPES, Camera as PyhapCamera
from pyhap.const import CATEGORY_CAMERA
from pyhap.util import callback as pyhap_callback
from homeassistant.components import camera
from homeassistant.components.ffmpeg import get_ffmpeg_manager
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.util.async_ import create_eager_task
from .accessories import TYPES, HomeDriver
from .const import (
    CHAR_MOTION_DETECTED,
    CONF_AUDIO_CODEC,
    CONF_AUDIO_MAP,
    CONF_AUDIO_PACKET_SIZE,
    CONF_LINKED_MOTION_SENSOR,
    CONF_MAX_FPS,
    CONF_MAX_HEIGHT,
    CONF_MAX_WIDTH,
    CONF_STREAM_ADDRESS,
    CONF_STREAM_COUNT,
    CONF_STREAM_SOURCE,
    CONF_SUPPORT_AUDIO,
    CONF_VIDEO_CODEC,
    CONF_VIDEO_MAP,
    CONF_VIDEO_PACKET_SIZE,
    CONF_VIDEO_PROFILE_NAMES,
    DEFAULT_AUDIO_CODEC,
    DEFAULT_AUDIO_MAP,
    DEFAULT_AUDIO_PACKET_SIZE,
    DEFAULT_MAX_FPS,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_WIDTH,
    DEFAULT_STREAM_COUNT,
    DEFAULT_SUPPORT_AUDIO,
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_MAP,
    DEFAULT_VIDEO_PACKET_SIZE,
    DEFAULT_VIDEO_PROFILE_NAMES,
    SERV_MOTION_SENSOR,
)
from .doorbell import HomeDoorbellAccessory
from .util import pid_is_alive, state_changed_event_is_same_state

_LOGGER = logging.getLogger(__name__)
VIDEO_OUTPUT: str = (
    "-map {v_map} -an -c:v {v_codec} {v_profile}-tune zerolatency -pix_fmt yuv420p -r {fps} "
    "-b:v {v_max_bitrate}k -bufsize {v_bufsize}k -maxrate {v_max_bitrate}k -payload_type 99 -ssrc {v_ssrc} "
    "-f rtp -srtp_out_suite AES_CM_128_HMAC_SHA1_80 -srtp_out_params {v_srtp_key} srtp://{address}:{v_port}"
    "?rtcpport={v_port}&localrtpport={v_port}&pkt_size={v_pkt_size}"
)
AUDIO_OUTPUT: str = (
    "-map {a_map} -vn -c:a {a_encoder} {a_application}-ac 1 -ar {a_sample_rate}k -b:a {a_max_bitrate}k "
    "-bufsize {a_bufsize}k -payload_type 110 -ssrc {a_ssrc} -f rtp -srtp_out_suite AES_CM_128_HMAC_SHA1_80 -srtp_out_params "
    "{a_srtp_key} srtp://{address}:{a_port}?rtcpport={a_port}&localrtpport={a_port}&pkt_size={a_pkt_size}"
)
SLOW_RESOLUTIONS: List[Tuple[int, int, int]] = [(320, 180, 15), (320, 240, 15)]
RESOLUTIONS: List[Tuple[int, int]] = [
    (320, 180),
    (320, 240),
    (480, 270),
    (480, 360),
    (640, 360),
    (640, 480),
    (1024, 576),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1920, 1080),
    (1600, 1200),
]
FFMPEG_WATCH_INTERVAL = timedelta(seconds=5)
FFMPEG_LOGGER: str = 'ffmpeg_logger'
FFMPEG_WATCHER: str = 'ffmpeg_watcher'
FFMPEG_PID: str = 'ffmpeg_pid'
SESSION_ID: str = 'session_id'
CONFIG_DEFAULTS: Dict[str, Any] = {
    CONF_SUPPORT_AUDIO: DEFAULT_SUPPORT_AUDIO,
    CONF_MAX_WIDTH: DEFAULT_MAX_WIDTH,
    CONF_MAX_HEIGHT: DEFAULT_MAX_HEIGHT,
    CONF_MAX_FPS: DEFAULT_MAX_FPS,
    CONF_AUDIO_CODEC: DEFAULT_AUDIO_CODEC,
    CONF_AUDIO_MAP: DEFAULT_AUDIO_MAP,
    CONF_VIDEO_MAP: DEFAULT_VIDEO_MAP,
    CONF_VIDEO_CODEC: DEFAULT_VIDEO_CODEC,
    CONF_VIDEO_PROFILE_NAMES: DEFAULT_VIDEO_PROFILE_NAMES,
    CONF_AUDIO_PACKET_SIZE: DEFAULT_AUDIO_PACKET_SIZE,
    CONF_VIDEO_PACKET_SIZE: DEFAULT_VIDEO_PACKET_SIZE,
    CONF_STREAM_COUNT: DEFAULT_STREAM_COUNT,
}


@TYPES.register("Camera")
class Camera(HomeDoorbellAccessory, PyhapCamera):
    def __init__(
        self,
        hass: HomeAssistant,
        driver: HomeDriver,
        name: str,
        entity_id: str,
        aid: int,
        config: Dict[str, Any],
    ) -> None:
        self._ffmpeg: Any = get_ffmpeg_manager(hass)
        for config_key, conf in CONFIG_DEFAULTS.items():
            if config_key not in config:
                config[config_key] = conf
        max_fps: int = config[CONF_MAX_FPS]
        max_width: int = config[CONF_MAX_WIDTH]
        max_height: int = config[CONF_MAX_HEIGHT]
        resolutions: List[Tuple[int, int, int]] = [
            (w, h, fps)
            for w, h, fps in SLOW_RESOLUTIONS
            if w <= max_width and h <= max_height and (fps < max_fps)
        ] + [
            (w, h, max_fps) for w, h in RESOLUTIONS if w <= max_width and h <= max_height
        ]
        video_options: Dict[str, Any] = {
            "codec": {
                "profiles": [
                    VIDEO_CODEC_PARAM_PROFILE_ID_TYPES["BASELINE"],
                    VIDEO_CODEC_PARAM_PROFILE_ID_TYPES["MAIN"],
                    VIDEO_CODEC_PARAM_PROFILE_ID_TYPES["HIGH"],
                ],
                "levels": [
                    VIDEO_CODEC_PARAM_LEVEL_TYPES["TYPE3_1"],
                    VIDEO_CODEC_PARAM_LEVEL_TYPES["TYPE3_2"],
                    VIDEO_CODEC_PARAM_LEVEL_TYPES["TYPE4_0"],
                ],
            },
            "resolutions": resolutions,
        }
        audio_options: Dict[str, Any] = {
            "codecs": [{"type": "OPUS", "samplerate": 24}, {"type": "OPUS", "samplerate": 16}]
        }
        stream_address: str = config.get(CONF_STREAM_ADDRESS, driver.state.address)
        options: Dict[str, Any] = {
            "video": video_options,
            "audio": audio_options,
            "address": stream_address,
            "srtp": True,
            "stream_count": config[CONF_STREAM_COUNT],
        }
        super().__init__(hass, driver, name, entity_id, aid, config, category=CATEGORY_CAMERA, options=options)
        self._char_motion_detected: Optional[Any] = None
        self.linked_motion_sensor: Optional[str] = self.config.get(CONF_LINKED_MOTION_SENSOR)
        self.motion_is_event: bool = False
        if self.linked_motion_sensor:
            self.motion_is_event = self.linked_motion_sensor.startswith("event.")
            if (state := self.hass.states.get(self.linked_motion_sensor)):
                serv_motion = self.add_preload_service(SERV_MOTION_SENSOR)
                self._char_motion_detected = serv_motion.configure_char(CHAR_MOTION_DETECTED, value=False)
                self._async_update_motion_state(None, state)

    @pyhap_callback
    @callback
    def run(self) -> None:
        if self._char_motion_detected:
            assert self.linked_motion_sensor is not None
            self._subscriptions.append(
                async_track_state_change_event(
                    self.hass, self.linked_motion_sensor, self._async_update_motion_state_event, job_type=Any
                )
            )
        super().run()

    @callback
    def _async_update_motion_state_event(self, event: Event) -> None:
        if not state_changed_event_is_same_state(event) and (new_state := event.data["new_state"]):
            self._async_update_motion_state(event.data.get("old_state"), new_state)

    @callback
    def _async_update_motion_state(self, old_state: Optional[State], new_state: State) -> None:
        state_str: str = new_state.state
        char = self._char_motion_detected
        assert char is not None
        if self.motion_is_event:
            if old_state is None or old_state.state == STATE_UNAVAILABLE or state_str in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                return
            _LOGGER.debug("%s: Set linked motion %s sensor to True/False", self.entity_id, self.linked_motion_sensor)
            char.set_value(True)
            char.set_value(False)
            return
        detected: bool = state_str == STATE_ON
        if char.value == detected:
            return
        char.set_value(detected)
        _LOGGER.debug("%s: Set linked motion %s sensor to %d", self.entity_id, self.linked_motion_sensor, detected)

    @callback
    def async_update_state(self, new_state: State) -> None:
        pass

    async def _async_get_stream_source(self) -> Optional[str]:
        stream_source: Optional[str] = self.config.get(CONF_STREAM_SOURCE)
        if stream_source:
            return stream_source
        try:
            stream_source = await camera.async_get_stream_source(self.hass, self.entity_id)
        except Exception:
            _LOGGER.exception(
                "Failed to get stream source - this could be a transient error or your camera might not be compatible with HomeKit yet"
            )
        return stream_source

    async def start_stream(self, session_info: Dict[str, Any], stream_config: Dict[str, Any]) -> bool:
        _LOGGER.debug("[%s] Starting stream with the following parameters: %s", session_info["id"], stream_config)
        input_source: Optional[str] = await self._async_get_stream_source()
        if not input_source:
            _LOGGER.error("Camera has no stream source")
            return False
        if "-i " not in input_source:
            input_source = "-i " + input_source
        video_profile: str = ""
        if self.config[CONF_VIDEO_CODEC] != "copy":
            video_profile = "-profile:v " + self.config[CONF_VIDEO_PROFILE_NAMES][int.from_bytes(stream_config["v_profile_id"], byteorder="big")] + " "
        audio_application: str = ""
        if self.config[CONF_AUDIO_CODEC] == "libopus":
            audio_application = "-application lowdelay "
        output_vars: Dict[str, Any] = stream_config.copy()
        output_vars.update(
            {
                "v_profile": video_profile,
                "v_bufsize": stream_config["v_max_bitrate"] * 4,
                "v_map": self.config[CONF_VIDEO_MAP],
                "v_pkt_size": self.config[CONF_VIDEO_PACKET_SIZE],
                "v_codec": self.config[CONF_VIDEO_CODEC],
                "a_bufsize": stream_config["a_max_bitrate"] * 4,
                "a_map": self.config[CONF_AUDIO_MAP],
                "a_pkt_size": self.config[CONF_AUDIO_PACKET_SIZE],
                "a_encoder": self.config[CONF_AUDIO_CODEC],
                "a_application": audio_application,
            }
        )
        output: str = VIDEO_OUTPUT.format(**output_vars)
        if self.config[CONF_SUPPORT_AUDIO]:
            output = output + " " + AUDIO_OUTPUT.format(**output_vars)
        _LOGGER.debug("FFmpeg output settings: %s", output)
        stream: HAFFmpeg = HAFFmpeg(self._ffmpeg.binary)
        opened: bool = await stream.open(
            cmd=[],
            input_source=input_source,
            output=output,
            extra_cmd="-hide_banner -nostats",
            stderr_pipe=True,
            stdout_pipe=False,
        )
        if not opened:
            _LOGGER.error("Failed to open ffmpeg stream")
            return False
        _LOGGER.debug("[%s] Started stream process - PID %d", session_info["id"], stream.process.pid)
        session_info["stream"] = stream
        session_info[FFMPEG_PID] = stream.process.pid
        stderr_reader: asyncio.StreamReader = await stream.get_reader(source=FFMPEG_STDERR)

        async def watch_session(_: Any) -> None:
            await self._async_ffmpeg_watch(session_info["id"])

        session_info[FFMPEG_LOGGER] = create_eager_task(self._async_log_stderr_stream(stderr_reader))
        session_info[FFMPEG_WATCHER] = async_track_time_interval(self.hass, watch_session, FFMPEG_WATCH_INTERVAL)
        return await self._async_ffmpeg_watch(session_info["id"])

    async def _async_log_stderr_stream(self, stderr_reader: asyncio.StreamReader) -> None:
        _LOGGER.debug("%s: ffmpeg: started", self.display_name)
        while True:
            line: bytes = await stderr_reader.readline()
            if line == b"":
                return
            _LOGGER.debug("%s: ffmpeg: %s", self.display_name, line.rstrip())

    async def _async_ffmpeg_watch(self, session_id: str) -> bool:
        ffmpeg_pid: int = self.sessions[session_id][FFMPEG_PID]
        if pid_is_alive(ffmpeg_pid):
            return True
        _LOGGER.warning("Streaming process ended unexpectedly - PID %d", ffmpeg_pid)
        self._async_stop_ffmpeg_watch(session_id)
        self.set_streaming_available(self.sessions[session_id]["stream_idx"])
        return False

    @callback
    def _async_stop_ffmpeg_watch(self, session_id: str) -> None:
        if FFMPEG_WATCHER not in self.sessions[session_id]:
            return
        self.sessions[session_id].pop(FFMPEG_WATCHER)()
        self.sessions[session_id].pop(FFMPEG_LOGGER).cancel()

    @callback
    def async_stop(self) -> None:
        for session_info in self.sessions.values():
            self.hass.async_create_background_task(self.stop_stream(session_info), "homekit.camera-stop-stream")
        super().async_stop()

    async def stop_stream(self, session_info: Dict[str, Any]) -> None:
        session_id: str = session_info["id"]
        stream: Optional[HAFFmpeg] = session_info.get("stream")
        if not stream:
            _LOGGER.debug("No stream for session ID %s", session_id)
            return
        self._async_stop_ffmpeg_watch(session_id)
        if not pid_is_alive(stream.process.pid):
            _LOGGER.warning("[%s] Stream already stopped", session_id)
            return
        for shutdown_method in ("close", "kill"):
            _LOGGER.debug("[%s] %s stream", session_id, shutdown_method)
            try:
                await getattr(stream, shutdown_method)()
            except Exception:
                _LOGGER.exception("[%s] Failed to %s stream", session_id, shutdown_method)
            else:
                return

    async def reconfigure_stream(self, session_info: Dict[str, Any], stream_config: Dict[str, Any]) -> bool:
        return True

    async def async_get_snapshot(self, image_size: Dict[str, int]) -> bytes:
        image = await camera.async_get_image(
            self.hass,
            self.entity_id,
            width=image_size["image-width"],
            height=image_size["image-height"],
        )
        return image.content
