"""Voice over IP (VoIP) implementation."""
from __future__ import annotations
import asyncio
from functools import partial
import logging
from pathlib import Path
import time
from typing import Optional, Union
from voip_utils import CallInfo, RtcpState, RtpDatagramProtocol, SdpInfo, VoipDatagramProtocol
from homeassistant.components.assist_pipeline import (
    Pipeline,
    PipelineNotFound,
    async_get_pipeline,
    select as pipeline_select,
)
from homeassistant.const import __version__
from homeassistant.core import HomeAssistant
from .const import CHANNELS, DOMAIN, RATE, RTP_AUDIO_SETTINGS, WIDTH

if TYPE_CHECKING:
    from .devices import VoIPDevices

_LOGGER = logging.getLogger(__name__)


def make_protocol(
    hass: HomeAssistant,
    devices: VoIPDevices,
    call_info: CallInfo,
    rtcp_state: Optional[RtcpState] = None,
) -> Union[VoipDatagramProtocol, PreRecordMessageProtocol]:
    """Plays a pre-recorded message if pipeline is misconfigured."""
    voip_device = devices.async_get_or_create(call_info)
    pipeline_id = pipeline_select.get_chosen_pipeline(hass, DOMAIN, voip_device.voip_id)
    try:
        pipeline: Optional[Pipeline] = async_get_pipeline(hass, pipeline_id)
    except PipelineNotFound:
        pipeline = None
    if pipeline is None or pipeline.stt_engine is None or pipeline.tts_engine is None:
        return PreRecordMessageProtocol(
            hass,
            'problem.pcm',
            opus_payload_type=call_info.opus_payload_type,
            rtcp_state=rtcp_state,
        )
    if (protocol := voip_device.protocol) is None:
        raise ValueError('VoIP satellite not found')
    protocol._rtp_input.opus_payload_type = call_info.opus_payload_type
    protocol._rtp_output.opus_payload_type = call_info.opus_payload_type
    protocol.rtcp_state = rtcp_state
    if protocol.rtcp_state is not None:
        protocol.rtcp_state.bye_callback = protocol.disconnect
    return protocol


class HassVoipDatagramProtocol(VoipDatagramProtocol):
    """HA UDP server for Voice over IP (VoIP)."""

    def __init__(self, hass: HomeAssistant, devices: VoIPDevices) -> None:
        """Set up VoIP call handler."""
        super().__init__(
            sdp_info=SdpInfo(
                username='homeassistant',
                id=time.monotonic_ns(),
                session_name='voip_hass',
                version=__version__,
            ),
            valid_protocol_factory=lambda call_info, rtcp_state: make_protocol(
                hass, devices, call_info, rtcp_state
            ),
            invalid_protocol_factory=lambda call_info, rtcp_state: PreRecordMessageProtocol(
                hass,
                'not_configured.pcm',
                opus_payload_type=call_info.opus_payload_type,
                rtcp_state=rtcp_state,
            ),
        )
        self.hass: HomeAssistant = hass
        self.devices: VoIPDevices = devices
        self._closed_event: asyncio.Event = asyncio.Event()

    def is_valid_call(self, call_info: CallInfo) -> bool:
        """Filter calls."""
        device = self.devices.async_get_or_create(call_info)
        return device.async_allow_call(self.hass)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Signal wait_closed when transport is completely closed."""
        self.hass.loop.call_soon_threadsafe(self._closed_event.set)

    async def wait_closed(self) -> None:
        """Wait for connection_lost to be called."""
        await self._closed_event.wait()


class PreRecordMessageProtocol(RtpDatagramProtocol):
    """Plays a pre-recorded message on a loop."""

    def __init__(
        self,
        hass: HomeAssistant,
        file_name: str,
        opus_payload_type: int,
        message_delay: float = 1.0,
        loop_delay: float = 2.0,
        rtcp_state: Optional[RtcpState] = None,
    ) -> None:
        """Set up RTP server."""
        super().__init__(
            rate=RATE,
            width=WIDTH,
            channels=CHANNELS,
            opus_payload_type=opus_payload_type,
            rtcp_state=rtcp_state,
        )
        self.hass: HomeAssistant = hass
        self.file_name: str = file_name
        self.message_delay: float = message_delay
        self.loop_delay: float = loop_delay
        self._audio_task: Optional[asyncio.Task[None]] = None
        self._audio_bytes: Optional[bytes] = None

    def on_chunk(self, audio_bytes: bytes) -> None:
        """Handle raw audio chunk."""
        if self.transport is None:
            return
        if self._audio_bytes is None:
            file_path = Path(__file__).parent / self.file_name
            self._audio_bytes = file_path.read_bytes()
        if self._audio_task is None:
            self._audio_task = self.hass.async_create_background_task(
                self._play_message(), 'voip_not_connected'
            )

    async def _play_message(self) -> None:
        await self.hass.async_add_executor_job(
            partial(
                self.send_audio,
                self._audio_bytes,
                silence_before=self.message_delay,
                **RTP_AUDIO_SETTINGS,
            )
        )
        await asyncio.sleep(self.loop_delay)
        self._audio_task = None
