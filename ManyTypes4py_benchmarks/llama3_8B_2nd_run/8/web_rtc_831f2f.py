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
from raiden.network.transport.matrix.utils import validate_and_parse_message
from raiden.utils.formatting import to_checksum_address
from raiden.utils.typing import Address, Any, Callable, Coroutine, Dict, List, Optional, Union
log = structlog.get_logger(__name__)

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

class _TaskHandler:
    def __init__(self) -> None:
        self._tasks: List[Task] = []
        self._greenlets: List[gevent.Greenlet] = []

    def schedule_task(self, coroutine: Callable[[], Coroutine[Union[Exception, Any], None, Any]], callback: Optional[Callable[[Any], None]] = None, *args: Any, **kwargs: Any) -> None:
        assert asyncio.iscoroutine(coroutine), 'must be a coroutine'
        task: Task = asyncio.create_task(coroutine)
        if callback is not None:
            assert callable(callback), 'must be a callable'

            def task_done(future: asyncio.Future) -> None:
                try:
                    result: Any = future.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    log.exception('Exception raised by task %r', task)
                else:
                    self.schedule_greenlet(callback, result, *args, **kwargs)
            task.add_done_callback(task_done)
        self._tasks.append(task)

    def schedule_greenlet(self, func: Optional[Callable[[], None]], *args: Any, **kwargs: Any) -> None:
        greenlet: gevent.Greenlet = gevent.spawn(func, *args, **kwargs)
        self._greenlets.append(greenlet)

    async def wait_for_tasks(self) -> None:
        tasks: List[Task] = self._tasks
        self._tasks = []
        for task in tasks:
            if not task.done() and not task.cancelled():
                task.cancel()
        pending_tasks: List[Task] = [task for task in tasks if not task.done()]
        logger: Optional[log.Logger] = getattr(self, 'log', log)
        logger.debug('Waiting for tasks', tasks=pending_tasks)
        try:
            return_values: List[Any] = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for value in return_values:
                if isinstance(value, Exception):
                    raise value
        except CancelledError:
            logger.debug('Pending tasks cancelled', cancelled=[task for task in tasks if task.cancelled()])

class _RTCConnection(_TaskHandler):
    def __init__(self, partner_address: Address, node_address: Address, signaling_send: Callable[[Address, str], None], ice_connection_closed: Callable[[], None], handle_message_callback: Callable[[ReceivedRaidenMessage], None]) -> None:
        super().__init__()
        self.node_address: Address = node_address
        self.partner_address: Address = partner_address
        self._closing_task: Optional[Task] = None
        self._call_id: str = self._make_call_id()
        self._signaling_send: Callable[[Address, str], None] = signaling_send
        self._ice_connection_closed: Callable[[], None] = ice_connection_closed
        self._handle_message_callback: Callable[[ReceivedRaidenMessage], None] = handle_message_callback
        self._aio_allow_candidates: asyncio.Event = asyncio.Event()
        self._aio_allow_remote_desc: asyncio.Event = asyncio.Event()
        self._channel: Optional[RTCDataChannel] = None
        self.log: log.Logger = log.bind(node=to_checksum_address(node_address), partner_address=to_checksum_address(partner_address))

    # ... rest of the class ...

class WebRTCManager(Runnable):
    def __init__(self, node_address: Address, process_messages: Callable[[List[ReceivedRaidenMessage]], None], signaling_send: Callable[[Address, str], None], stop_event: asyncio.Event) -> None:
        super().__init__()
        self.node_address: Address = node_address
        self._process_messages: Callable[[List[ReceivedRaidenMessage]], None] = process_messages
        self._signaling_send: Callable[[Address, str], None] = signaling_send
        self._stop_event: asyncio.Event = stop_event
        self._address_to_connection: Dict[Address, _RTCConnection] = {}
        self._address_to_lock: Dict[Address, Semaphore] = {}
        self.log: log.Logger = log.bind(node=to_checksum_address(node_address))

    # ... rest of the class ...
