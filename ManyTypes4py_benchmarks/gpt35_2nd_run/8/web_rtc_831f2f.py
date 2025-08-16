from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

class _RTCMessageType(Enum):
    OFFER: str = 'offer'
    ANSWER: str = 'answer'
    CANDIDATES: str = 'candidates'
    HANGUP: str = 'hangup'

class _SDPTypes(Enum):
    OFFER: str = 'offer'
    ANSWER: str = 'answer'

class _RTCChannelState(Enum):
    CONNECTING: str = 'connecting'
    OPEN: str = 'open'
    CLOSING: str = 'closing'
    CLOSED: str = 'closed'

class _RTCSignallingState(Enum):
    STABLE: str = 'stable'
    HAVE_LOCAL_OFFER: str = 'have-local-offer'
    HAVE_REMOTE_OFFER: str = 'have-remote-offer'
    CLOSED: str = 'closed'

class _ConnectionState(Enum):
    NEW: str = 'new'
    CONNECTING: str = 'connecting'
    CONNECTED: str = 'connected'
    FAILED: str = 'failed'
    CLOSED: str = 'closed'

_SDP_MID_DEFAULT: str = '0'
_SDP_MLINE_INDEX_DEFAULT: int = 0

class _TaskHandler:

    def __init__(self) -> None:
        self._tasks: List[Task] = []
        self._greenlets: List[gevent.Greenlet] = []

    def schedule_task(self, coroutine: Coroutine, callback: Optional[Callable], *args, **kwargs) -> None:
        ...

    def schedule_greenlet(self, func: Optional[Callable], *args, **kwargs) -> None:
        ...

    async def wait_for_tasks(self) -> Any:
        ...

class _RTCConnection(_TaskHandler):

    def __init__(self, partner_address: Address, node_address: Address, signaling_send: Callable, ice_connection_closed: Callable, handle_message_callback: Callable) -> None:
        ...

    @staticmethod
    def from_offer(partner_address: Address, node_address: Address, signaling_send: Callable, ice_connection_closed: Callable, handle_message_callback: Callable, offer: Dict[str, Any]) -> '_RTCConnection':
        ...

    @property
    def initiator_address(self) -> Address:
        ...

    def _setup_peer_connection(self) -> None:
        ...

    def _set_channel_callbacks(self) -> None:
        ...

    def channel_open(self) -> bool:
        ...

    @property
    def call_id(self) -> str:
        ...

    async def _try_signaling(self, coroutine: Coroutine) -> Any:
        ...

    async def _set_local_description(self, description: RTCSessionDescription) -> None:
        ...

    def _make_call_id(self) -> str:
        ...

    async def _initialize_signaling(self) -> None:
        ...

    def initialize_signaling(self) -> None:
        ...

    async def _process_signaling(self, description: Dict[str, Any]) -> Any:
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

    def close(self) -> Task:
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

class WebRTCManager(Runnable):

    def __init__(self, node_address: Address, process_messages: Callable, signaling_send: Callable, stop_event: GEvent) -> None:
        ...

    def get_lock(self, address: Address) -> Semaphore:
        ...

    def is_locked(self, address: Address) -> bool:
        ...

    def _handle_message(self, message_data: str, partner_address: Address) -> None:
        ...

    def _handle_ice_connection_closed(self, conn: _RTCConnection) -> None:
        ...

    def _wrapped_initialize_web_rtc(self, address: Address) -> None:
        ...

    def _initialize_web_rtc(self, partner_address: Address) -> None:
        ...

    def get_channel_init_timeout(self) -> float:
        ...

    def _add_connection(self, partner_address: Address, conn: _RTCConnection) -> None:
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
