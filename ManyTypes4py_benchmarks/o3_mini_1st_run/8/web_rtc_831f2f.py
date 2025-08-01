#!/usr/bin/env python3
import asyncio
import json
import time
from asyncio import CancelledError, Task
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar
import gevent
from gevent.event import Event as GEvent
from gevent.lock import Semaphore
import structlog
from aiortc import InvalidStateError, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from raiden.network.transport.matrix.client import ReceivedRaidenMessage
from raiden.network.transport.matrix.rtc.aiogevent import wrap_greenlet, yield_future
from raiden.network.transport.matrix.utils import validate_and_parse_message
from raiden.utils.formatting import to_checksum_address
from raiden.utils.runnable import Runnable
from raiden.utils.typing import Address

log = structlog.get_logger(__name__)

# Type variable for _try_signaling
T = TypeVar("T")

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

_SDP_MID_DEFAULT = '0'
_SDP_MLINE_INDEX_DEFAULT = 0

class _TaskHandler:
    def __init__(self) -> None:
        self._tasks: List[asyncio.Task[Any]] = []
        self._greenlets: List[gevent.Greenlet] = []

    def schedule_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
        callback: Optional[Callable[..., Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        assert asyncio.iscoroutine(coroutine), 'must be a coroutine'
        task: asyncio.Task[Any] = asyncio.create_task(coroutine)
        if callback is not None:
            assert callable(callback), 'must be a callable'

            def task_done(future: asyncio.Future[Any]) -> None:
                try:
                    result = future.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    log.exception('Exception raised by task %r', task)
                else:
                    self.schedule_greenlet(callback, result, *args, **kwargs)
            task.add_done_callback(task_done)
        self._tasks.append(task)

    def schedule_greenlet(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        greenlet = gevent.spawn(func, *args, **kwargs)
        self._greenlets.append(greenlet)

    async def wait_for_tasks(self) -> Any:
        tasks = self._tasks
        self._tasks = []
        for task in tasks:
            if not task.done() and (not task.cancelled()):
                task.cancel()
        pending_tasks = [task for task in tasks if not task.done()]
        logger = getattr(self, 'log', log)
        logger.debug('Waiting for tasks', tasks=pending_tasks)
        try:
            return_values = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for value in return_values:
                if isinstance(value, Exception):
                    raise value
        except CancelledError:
            logger.debug('Pending tasks cancelled', cancelled=[task for task in tasks if task.cancelled()])

class _RTCConnection(_TaskHandler):
    def __init__(
        self,
        partner_address: Address,
        node_address: Address,
        signaling_send: Callable[[Address, str], None],
        ice_connection_closed: Callable[["_RTCConnection"], None],
        handle_message_callback: Callable[[Any, Address], None]
    ) -> None:
        super().__init__()
        self.node_address: Address = node_address
        self.partner_address: Address = partner_address
        self._closing_task: Optional[asyncio.Task[Any]] = None
        self._call_id: str = self._make_call_id()
        self._signaling_send: Callable[[Address, str], None] = signaling_send
        self._ice_connection_closed: Callable[["_RTCConnection"], None] = ice_connection_closed
        self._handle_message_callback: Callable[[Any, Address], None] = handle_message_callback
        self._aio_allow_candidates: asyncio.Event = asyncio.Event()
        self._aio_allow_remote_desc: asyncio.Event = asyncio.Event()
        self._channel: Optional[RTCDataChannel] = None
        self._initiator_address: Address = node_address
        self.log = log.bind(node=to_checksum_address(node_address), partner_address=to_checksum_address(partner_address))
        self._setup_peer_connection()

    def _setup_peer_connection(self) -> None:
        self.peer_connection: RTCPeerConnection = RTCPeerConnection()
        self.peer_connection.on('icegatheringstatechange', self._on_ice_gathering_state_change)
        self.peer_connection.on('signalingstatechange', self._on_signalling_state_change)
        self.peer_connection.on('connectionstatechange', self._on_connection_state_change)

    @staticmethod
    def from_offer(
        partner_address: Address,
        node_address: Address,
        signaling_send: Callable[[Address, str], None],
        ice_connection_closed: Callable[["_RTCConnection"], None],
        handle_message_callback: Callable[[Any, Address], None],
        offer: Dict[str, Any]
    ) -> "_RTCConnection":
        conn = _RTCConnection(partner_address, node_address, signaling_send, ice_connection_closed, handle_message_callback)
        conn._call_id = offer['call_id']
        conn._initiator_address = partner_address
        return conn

    @property
    def initiator_address(self) -> Address:
        return self._initiator_address

    def _set_channel_callbacks(self) -> None:
        assert self._channel is not None, 'must be set'
        self._channel.on('message', self._on_channel_message)
        self._channel.on('open', self._on_channel_open)

    def channel_open(self) -> bool:
        return self._channel is not None and self._channel.readyState == _RTCChannelState.OPEN.value

    @property
    def call_id(self) -> str:
        return self._call_id

    async def _try_signaling(self, coroutine: Coroutine[Any, Any, T]) -> Optional[T]:
        try:
            return await coroutine
        except InvalidStateError:
            self.log.debug(
                'Invalid connection state',
                signaling_state=self.peer_connection.signalingState,
                ice_connection_state=self.peer_connection.iceConnectionState
            )
        except AttributeError:
            self.log.exception('Attribute error in coroutine', coroutine=coroutine)
        self.close()
        return None

    async def _set_local_description(self, description: RTCSessionDescription) -> None:
        self.log.debug('Set local description', description=description)
        await self._try_signaling(self.peer_connection.setLocalDescription(description))
        self._aio_allow_remote_desc.set()

    def _make_call_id(self) -> str:
        timestamp = time.time()
        address1, address2 = (self.node_address, self.partner_address)
        return f'{to_checksum_address(address1)}|{to_checksum_address(address2)}|{timestamp}'

    async def _initialize_signaling(self) -> Optional[RTCSessionDescription]:
        """Coroutine to create channel. Setting up channel in aiortc."""
        if self._closing_task is not None:
            return None
        self._channel = self.peer_connection.createDataChannel(self.call_id)
        self._set_channel_callbacks()
        offer = await self._try_signaling(self.peer_connection.createOffer())
        if offer is None:
            return None
        self.log.debug('Created WebRTC offer', offer=offer)
        self.schedule_task(self._set_local_description(offer))
        return offer

    def initialize_signaling(self) -> None:
        self.schedule_task(self._initialize_signaling(), callback=self._handle_sdp_callback)

    async def _process_signaling(self, description: Dict[str, Any]) -> Optional[RTCSessionDescription]:
        remote_description = RTCSessionDescription(description['sdp'], description['type'])
        sdp_type: str = description['type']
        if self._closing_task is not None:
            return None
        if self._initiator_address == self.node_address:
            await self._aio_allow_remote_desc.wait()
        self.log.debug('Set Remote Description', description=description)
        await self._try_signaling(self.peer_connection.setRemoteDescription(remote_description))
        if self.peer_connection.remoteDescription is None:
            return None
        if sdp_type == _SDPTypes.ANSWER.value:
            return None
        self.peer_connection.on('datachannel', self._on_datachannel)
        answer = await self._try_signaling(self.peer_connection.createAnswer())
        if answer is None:
            return None
        self.schedule_task(self._set_local_description(answer))
        return answer

    def process_signaling(self, description: Dict[str, Any]) -> None:
        self.schedule_task(self._process_signaling(description), callback=self._handle_sdp_callback)

    async def _set_candidates(self, content: Dict[str, Any]) -> None:
        if self.peer_connection.sctp is None:
            await self._aio_allow_candidates.wait()
        assert self.peer_connection.sctp, 'SCTP should be set by now'
        for candidate in content['candidates']:
            rtc_ice_candidate = candidate_from_sdp(candidate['candidate'])
            rtc_ice_candidate.sdpMid = candidate['sdpMid']
            rtc_ice_candidate.sdpMLineIndex = candidate['sdpMLineIndex']
            if rtc_ice_candidate.sdpMid != self.peer_connection.sctp.mid:
                self.log.debug('Invalid candidate. Wrong sdpMid', candidate=candidate, sctp_sdp_mid=self.peer_connection.sctp.mid)
                continue
            await self.peer_connection.addIceCandidate(rtc_ice_candidate)

    def set_candidates(self, content: Dict[str, Any]) -> None:
        self.schedule_task(self._set_candidates(content))

    async def _send_message(self, message: str) -> None:
        """Sends message through aiortc. Not an async function. Output is written to buffer."""
        if self._channel is not None and self._channel.readyState == _RTCChannelState.OPEN.value:
            self.log.debug('Sending message in asyncio kingdom', channel=self._channel.label, message=message, time=time.time())
            self._channel.send(message)
            try:
                await self.peer_connection.sctp._transmit()
                await self.peer_connection.sctp._data_channel_flush()
                await self.peer_connection.sctp._transmit()
            except ConnectionError:
                self.log.debug('Connection error occurred while trying to send message')
                self.close()
                return
        else:
            self.log.debug(
                'Channel is not open but trying to send a message.',
                ready_state=self._channel.readyState if self._channel is not None else 'No channel exists'
            )

    def send_message(self, message: str) -> None:
        self.schedule_task(self._send_message(message))

    async def _close(self) -> None:
        self.log.debug('Closing peer connection')
        await self.wait_for_tasks()
        await wrap_greenlet(gevent.spawn(gevent.killall, self._greenlets))
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        await self.peer_connection.close()
        self.peer_connection = None  # type: ignore
        self._ice_connection_closed(self)

    def close(self) -> Optional[asyncio.Task[Any]]:
        if self._closing_task is None:
            self._closing_task = asyncio.create_task(self._close())
        return self._closing_task

    def _handle_candidates_callback(self, candidates: List[Dict[str, Any]]) -> None:
        message: Dict[str, Any] = {'type': _RTCMessageType.CANDIDATES.value, 'candidates': candidates, 'call_id': self.call_id}
        self._signaling_send(self.partner_address, json.dumps(message))

    def send_hangup_message(self) -> None:
        hangup_message: Dict[str, Any] = {'type': _RTCMessageType.HANGUP.value, 'call_id': self.call_id}
        self._signaling_send(self.partner_address, json.dumps(hangup_message))

    def _handle_sdp_callback(self, rtc_session_description: Optional[RTCSessionDescription]) -> None:
        """
        This is a callback function to process SDP (Session Description Protocol) messages.
        These messages are part of the ROAP (RTC Offer Answer Protocol) which is also called
        signalling. Messages are exchanged via Matrix.
        Args:
            rtc_session_description: sdp message for the partner
        """
        if rtc_session_description is None:
            return
        sdp_type = rtc_session_description.type
        message: Dict[str, Any] = {'type': sdp_type, 'sdp': rtc_session_description.sdp, 'call_id': self.call_id}
        self.log.debug(f'Send {sdp_type} to partner', partner_address=to_checksum_address(self.partner_address), sdp_description=message)
        self._signaling_send(self.partner_address, json.dumps(message))

    def __repr__(self) -> str:
        return f'<_RTCConnection[{hex(id(self))}] {self.call_id}>'

    def _on_datachannel(self, channel: RTCDataChannel) -> None:
        self._channel = channel
        self._on_channel_open()
        self._set_channel_callbacks()

    def _on_channel_open(self) -> None:
        assert self._channel is not None, 'must be set'
        self.log.debug('WebRTC data channel open', node=to_checksum_address(self.node_address), label=self._channel.label)

    def _on_channel_message(self, message: Any) -> None:
        assert self._channel is not None, 'channel not set but received message'
        self.log.debug('Received message in asyncio kingdom', channel=self._channel.label, message=message, time=time.time())
        self.schedule_greenlet(self._handle_message_callback, message_data=message, partner_address=self.partner_address)

    def _on_ice_gathering_state_change(self) -> None:
        self.log.debug('ICE gathering state changed', state=self.peer_connection.iceGatheringState)
        if self.peer_connection.iceGatheringState != 'complete':
            return
        rtc_ice_candidates = self.peer_connection.sctp.transport.transport.iceGatherer.getLocalCandidates()
        candidates: List[Dict[str, Any]] = []
        for candidate in rtc_ice_candidates:
            candidate_dict = {
                'candidate': f'candidate:{candidate_to_sdp(candidate)}',
                'sdpMid': candidate.sdpMid if candidate.sdpMid is not None else _SDP_MID_DEFAULT,
                'sdpMLineIndex': candidate.sdpMLineIndex if candidate.sdpMLineIndex is not None else _SDP_MLINE_INDEX_DEFAULT
            }
            candidates.append(candidate_dict)
        self.schedule_greenlet(self._handle_candidates_callback, candidates=candidates)

    def _on_connection_state_change(self) -> None:
        connection_state: str = self.peer_connection.connectionState
        self.log.debug('Connection state changed', connection_state=connection_state)
        if connection_state in (_ConnectionState.CLOSED.value, _ConnectionState.FAILED.value):
            self.close()

    def _on_signalling_state_change(self) -> None:
        signaling_state: str = self.peer_connection.signalingState
        self.log.debug('Signaling state changed', signaling_state=signaling_state)
        if signaling_state in (_RTCSignallingState.HAVE_REMOTE_OFFER.value, _RTCSignallingState.CLOSED.value):
            self._aio_allow_candidates.set()

class WebRTCManager(Runnable):
    def __init__(
        self,
        node_address: Address,
        process_messages: Callable[[List[ReceivedRaidenMessage]], None],
        signaling_send: Callable[[Address, str], None],
        stop_event: GEvent
    ) -> None:
        super().__init__()
        self.node_address: Address = node_address
        self._process_messages: Callable[[List[ReceivedRaidenMessage]], None] = process_messages
        self._signaling_send: Callable[[Address, str], None] = signaling_send
        self._stop_event: GEvent = stop_event
        self._address_to_connection: Dict[Address, _RTCConnection] = {}
        self._address_to_lock: Dict[Address, Semaphore] = {}
        self.log = log.bind(node=to_checksum_address(node_address))

    def get_lock(self, address: Address) -> Semaphore:
        if address not in self._address_to_lock:
            self._address_to_lock[address] = Semaphore()
        return self._address_to_lock[address]

    def is_locked(self, address: Address) -> bool:
        return self.get_lock(address).locked()

    def _handle_message(self, message_data: Any, partner_address: Address) -> None:
        messages: List[ReceivedRaidenMessage] = []
        for msg in validate_and_parse_message(message_data, partner_address):
            messages.append(ReceivedRaidenMessage(message=msg, sender=partner_address))
        self._process_messages(messages)

    def _handle_ice_connection_closed(self, conn: _RTCConnection) -> None:
        self._address_to_connection.pop(conn.partner_address, None)
        if conn.initiator_address == self.node_address:
            self.health_check(conn.partner_address)

    def _wrapped_initialize_web_rtc(self, address: Address) -> None:
        attempt = 0
        while attempt < 3 and (not self.has_ready_channel(address)):
            self._initialize_web_rtc(address)
            attempt += 1

    def _initialize_web_rtc(self, partner_address: Address) -> None:
        if partner_address in self._address_to_connection:
            return
        if self._stop_event.is_set():
            return
        self.log.debug('Establishing WebRTC channel', partner_address=to_checksum_address(partner_address))
        if self.is_locked(partner_address):
            return
        conn = _RTCConnection(partner_address, self.node_address, self._signaling_send, self._handle_ice_connection_closed, self._handle_message)
        self._add_connection(partner_address, conn)
        conn.initialize_signaling()
        if self._stop_event.wait(timeout=self.get_channel_init_timeout()):
            return
        if conn is not self._address_to_connection.get(partner_address, None):
            return
        if not self.has_ready_channel(partner_address):
            self.log.debug('Could not establish channel', partner_address=to_checksum_address(partner_address))
            conn.send_hangup_message()
            with self.get_lock(partner_address):
                self.close_connection(partner_address)

    def get_channel_init_timeout(self) -> float:
        """Returns the number of seconds to wait for a channel to be established."""
        return 30.0

    def _add_connection(self, partner_address: Address, conn: _RTCConnection) -> None:
        assert partner_address not in self._address_to_connection, 'must not be there already'
        self._address_to_connection[partner_address] = conn

    def has_ready_channel(self, partner_address: Address) -> bool:
        conn = self._address_to_connection.get(partner_address)
        return conn is not None and conn.channel_open()

    def _reset_state(self) -> None:
        self._address_to_connection = {}

    def _set_candidates_for_address(self, partner_address: Address, content: Dict[str, Any]) -> None:
        conn = self._address_to_connection.get(partner_address)
        if conn is not None:
            conn.set_candidates(content)

    def _process_signaling_for_address(self, partner_address: Address, rtc_message_type: str, description: Dict[str, Any]) -> None:
        conn = self._address_to_connection.get(partner_address)
        if rtc_message_type == _RTCMessageType.OFFER.value:
            if conn is not None:
                if conn.call_id < description['call_id']:
                    self.close_connection(partner_address)
                else:
                    return
            if self._stop_event.is_set():
                return
            conn = _RTCConnection.from_offer(partner_address, self.node_address, self._signaling_send, self._handle_ice_connection_closed, self._handle_message, description)
            self._add_connection(partner_address, conn)
        elif conn is None:
            return
        conn.process_signaling(description)

    def send_message(self, partner_address: Address, message: str) -> None:
        conn = self._address_to_connection[partner_address]
        conn.send_message(message)

    def health_check(self, partner_address: Address) -> None:
        if partner_address in self._address_to_connection:
            return
        self._schedule_new_greenlet(self._wrapped_initialize_web_rtc, partner_address)  # type: ignore

    def close_connection(self, partner_address: Address) -> None:
        conn = self._address_to_connection.get(partner_address)
        if conn is not None:
            yield_future(conn.close())

    def _process_signaling_message(self, partner_address: Address, rtc_message_type: str, content: Dict[str, Any]) -> None:
        if rtc_message_type in [_RTCMessageType.OFFER.value, _RTCMessageType.ANSWER.value] and 'sdp' in content:
            with self.get_lock(partner_address):
                self._process_signaling_for_address(partner_address, rtc_message_type, content)
        elif rtc_message_type == _RTCMessageType.HANGUP.value:
            self.close_connection(partner_address)
        elif rtc_message_type == _RTCMessageType.CANDIDATES.value:
            self._set_candidates_for_address(partner_address, content)
        else:
            self.log.error('Unknown WebRTC message type', partner_address=to_checksum_address(partner_address), type=rtc_message_type)

    def process_signaling_message(self, partner_address: Address, rtc_message_type: str, content: Dict[str, Any]) -> None:
        self._schedule_new_greenlet(self._process_signaling_message, partner_address, rtc_message_type, content)  # type: ignore

    def stop(self) -> None:
        self.log.debug('Closing WebRTC connections')
        for conn in tuple(self._address_to_connection.values()):
            conn.send_hangup_message()
        for partner_address in list(self._address_to_connection.keys()):
            self.close_connection(partner_address)
        gevent.killall(self.greenlets)  # type: ignore
        self._reset_state()
