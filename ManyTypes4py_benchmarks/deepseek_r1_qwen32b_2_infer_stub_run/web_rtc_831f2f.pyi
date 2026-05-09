import asyncio
import json
import time
from asyncio import CancelledError, Task
from enum import Enum
import gevent
import structlog
from aiortc import InvalidStateError, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from gevent.event import Event as GEvent
from gevent.lock import Semaphore
from raiden.network.transport.matrix.client import ReceivedRaidenMessage
from raiden.network.transport.matrix.rtc.aiogevent import wrap_greenlet, yield_future
from raiden.utils.formatting import to_checksum_address
from raiden.utils.typing import Address, Any, Callable, Coroutine, Dict, List, Optional, Union

log = structlog.get_logger(__name__)

class _RTCMessageType(Enum):
    OFFER = 'offer'
    ANSWER = 'answer'
    CANDIDATES = 'candidates'
    HANGUP = 'hangup'

class _SDPTypes(Enum):
    OFFER = 'offer'
    ANSWER = 'answer'

class _RTCChannelState(Enum):
    CONNECTING = 'connecting'
    OPEN = 'open'
    CLOSING = 'closing'
    CLOSED = 'closed'

class _RTCSignallingState(Enum):
    STABLE = 'stable'
    HAVE_LOCAL_OFFER = 'have-local-offer'
    HAVE_REMOTE_OFFER = 'have-remote-offer'
    CLOSED = 'closed'

class _ConnectionState(Enum):
    NEW = 'new'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    FAILED = 'failed'
    CLOSED = 'closed'

class _TaskHandler:
    def __init__(self) -> None:
        ...
    
    def schedule_task(self, coroutine: Coroutine, callback: Optional[Callable[..., None]] = None, *args: Any, **kwargs: Any) -> None:
        ...
    
    def schedule_greenlet(self, func: Optional[Callable[..., None]] = None, *args: Any, **kwargs: Any) -> None:
        ...
    
    async def wait_for_tasks(self) -> None:
        ...

class _RTCConnection(_TaskHandler):
    def __init__(self, partner_address: Address, node_address: Address, signaling_send: Callable[[Address, str], None], ice_connection_closed: Callable[[ '_RTCConnection'], None], handle_message_callback: Callable[[Dict[str, Any]], None]) -> None:
        ...
    
    @staticmethod
    def from_offer(partner_address: Address, node_address: Address, signaling_send: Callable[[Address, str], None], ice_connection_closed: Callable[[ '_RTCConnection'], None], handle_message_callback: Callable[[Dict[str, Any]], None], offer: Dict[str, Any]) -> '_RTCConnection':
        ...
    
    def _set_channel_callbacks(self) -> None:
        ...
    
    def channel_open(self) -> bool:
        ...
    
    @property
    def initiator_address(self) -> Address:
        ...
    
    @property
    def call_id(self) -> str:
        ...
    
    async def _try_signaling(self, coroutine: Coroutine) -> Optional[Any]:
        ...
    
    async def _set_local_description(self, description: RTCSessionDescription) -> None:
        ...
    
    def _make_call_id(self) -> str:
        ...
    
    async def _initialize_signaling(self) -> Optional[Dict[str, Any]]:
        ...
    
    def initialize_signaling(self) -> None:
        ...
    
    async def _process_signaling(self, description: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...
    
    def process_signaling(self, description: Dict[str, Any]) -> None:
        ...
    
    async def _set_candidates(self, content: Dict[str, Any]) -> None:
        ...
    
    def set_candidates(self, content: Dict[str, Any]) -> None:
        ...
    
    async def _send_message(self, message: str) -> None:
        ...
    
    def send_message(self, message: str) -> None:
        ...
    
    async def _close(self) -> None:
        ...
    
    def close(self) -> Optional[Task]:
        ...
    
    def _handle_candidates_callback(self, candidates: List[Dict[str, Any]]) -> None:
        ...
    
    def send_hangup_message(self) -> None:
        ...
    
    def _handle_sdp_callback(self, rtc_session_description: RTCSessionDescription) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def _on_datachannel(self, channel: RTCDataChannel) -> None:
        ...
    
    def _on_channel_open(self) -> None:
        ...
    
    def _on_channel_message(self, message: str) -> None:
        ...
    
    def _on_ice_gathering_state_change(self) -> None:
        ...
    
    def _on_connection_state_change(self) -> None:
        ...
    
    def _on_signalling_state_change(self) -> None:
        ...

class WebRTCManager:
    def __init__(self, node_address: Address, process_messages: Callable[[List[ReceivedRaidenMessage]], None], signaling_send: Callable[[Address, str], None], stop_event: GEvent) -> None:
        ...
    
    def get_lock(self, address: Address) -> Semaphore:
        ...
    
    def is_locked(self, address: Address) -> bool:
        ...
    
    def _handle_message(self, message_data: str, partner_address: Address) -> None:
        ...
    
    def _handle_ice_connection_closed(self, conn: '_RTCConnection') -> None:
        ...
    
    def _wrapped_initialize_web_rtc(self, address: Address) -> None:
        ...
    
    def _initialize_web_rtc(self, partner_address: Address) -> None:
        ...
    
    def get_channel_init_timeout(self) -> float:
        ...
    
    def _add_connection(self, partner_address: Address, conn: '_RTCConnection') -> None:
        ...
    
    def has_ready_channel(self, partner_address: Address) -> bool:
        ...
    
    def _reset_state(self) -> None:
        ...
    
    def _set_candidates_for_address(self, partner_address: Address, content: Dict[str, Any]) -> None:
        ...
    
    def _process_signaling_for_address(self, partner_address: Address, rtc_message_type: str, description: Dict[str, Any]) -> None:
        ...
    
    def send_message(self, partner_address: Address, message: str) -> None:
        ...
    
    def health_check(self, partner_address: Address) -> None:
        ...
    
    def close_connection(self, partner_address: Address) -> None:
        ...
    
    def _process_signaling_message(self, partner_address: Address, rtc_message_type: str, content: Dict[str, Any]) -> None:
        ...
    
    def process_signaling_message(self, partner_address: Address, rtc_message_type: str, content: Dict[str, Any]) -> None:
        ...
    
    def stop(self) -> None:
        ...