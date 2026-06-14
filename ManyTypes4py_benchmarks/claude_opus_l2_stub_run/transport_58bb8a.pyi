import asyncio
import itertools
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from json import JSONDecodeError
from typing import TYPE_CHECKING, Counter as CounterType, Generator
from uuid import UUID

import gevent
import structlog
from gevent.event import Event
from gevent.queue import JoinableQueue
from web3.types import BlockIdentifier

from raiden.constants import (
    CommunicationMedium,
    DeviceIDs,
    Environment,
    MatrixMessageType,
)
from raiden.messages.abstract import Message, RetrieableMessage, SignedRetrieableMessage
from raiden.messages.healthcheck import Ping, Pong
from raiden.messages.synchronization import Delivered, Processed
from raiden.network.pathfinding import PFSProxy
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.transport.matrix.client import (
    GMatrixClient,
    MatrixMessage,
    ReceivedCallMessage,
    ReceivedRaidenMessage,
)
from raiden.network.transport.matrix.rtc.web_rtc import WebRTCManager
from raiden.network.transport.matrix.utils import (
    DisplayNameCache,
    MessageAckTimingKeeper,
    UserPresence,
)
from raiden.settings import MatrixTransportConfig
from raiden.transfer.identifiers import QueueIdentifier
from raiden.transfer.state import QueueIdsToQueues
from raiden.utils.runnable import Runnable
from raiden.utils.typing import (
    Address,
    AddressHex,
    AddressMetadata,
    Any,
    Callable,
    ChainID,
    Dict,
    Iterable,
    Iterator,
    List,
    MessageID,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    UserID,
)

if TYPE_CHECKING:
    from raiden.raiden_service import RaidenService

log: structlog.BoundLogger
RETRY_QUEUE_IDLE_AFTER: int
SET_PRESENCE_INTERVAL: int

@dataclass
class MessagesQueue:
    queue_identifier: QueueIdentifier
    messages: List[Tuple[Message, Optional[AddressMetadata]]]

def _metadata_key_func(message_data: _RetryQueue._MessageData) -> str: ...

class _RetryQueue(Runnable):
    """A helper Runnable to send batched messages to receiver through transport"""

    class _MessageData(NamedTuple):
        queue_identifier: QueueIdentifier
        message: Message
        text: str
        expiration_generator: Generator[bool, None, None]
        address_metadata: Optional[AddressMetadata]

    transport: MatrixTransport
    receiver: Address
    _message_queue: List[_MessageData]
    _notify_event: Event
    _idle_since: int

    def __init__(self, transport: MatrixTransport, receiver: Address) -> None: ...

    @property
    def log(self) -> Any: ...

    @staticmethod
    def _expiration_generator(
        timeout_generator: Iterator[float], now: Callable[[], float] = ...
    ) -> Generator[bool, None, None]: ...

    def enqueue(
        self,
        queue_identifier: QueueIdentifier,
        messages: List[Tuple[Message, Optional[AddressMetadata]]],
    ) -> None: ...

    def enqueue_unordered(
        self, message: Message, address_metadata: Optional[AddressMetadata] = ...
    ) -> None: ...

    def notify(self) -> None: ...
    def _check_and_send(self) -> None: ...
    def _run(self) -> None: ...

    @property
    def is_idle(self) -> bool: ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class MatrixTransport(Runnable):
    log: Any
    _uuid: UUID
    _config: MatrixTransportConfig
    _environment: Environment
    _raiden_service: Optional[RaidenService]
    _web_rtc_manager: Optional[WebRTCManager]
    _client: GMatrixClient
    server_url: str
    _server_name: str
    _all_server_names: Set[str]
    greenlets: List[gevent.Greenlet]
    _address_to_retrier: Dict[Address, _RetryQueue]
    _displayname_cache: DisplayNameCache
    _broadcast_queue: JoinableQueue
    _started: bool
    _stop_event: Event
    _broadcast_event: Event
    _prioritize_broadcast_messages: bool
    _counters: Dict[str, Counter]
    _message_timing_keeper: Optional[MessageAckTimingKeeper]
    services_addresses: Dict[Address, int]

    def __init__(
        self,
        config: MatrixTransportConfig,
        environment: Environment,
        enable_tracing: bool = ...,
    ) -> None: ...

    @property
    def started(self) -> bool: ...

    def __repr__(self) -> str: ...

    @property
    def checksummed_address(self) -> Optional[str]: ...

    @property
    def _node_address(self) -> Optional[Address]: ...

    @property
    def user_id(self) -> Optional[UserID]: ...

    @property
    def displayname(self) -> str: ...

    @property
    def address_metadata(self) -> Optional[Dict[str, Any]]: ...

    def start(
        self, raiden_service: RaidenService, prev_auth_data: Optional[str]
    ) -> None: ...

    def health_check_web_rtc(self, partner: Address) -> None: ...
    def _set_presence(self, state: UserPresence) -> None: ...
    def _run(self) -> None: ...
    def stop(self) -> None: ...
    def send_async(self, message_queues: List[MessagesQueue]) -> None: ...
    def update_services_addresses(self, addresses_validity: Dict[Address, int]) -> None: ...
    def expire_services_addresses(
        self, current_timestamp: int, block_number: int
    ) -> None: ...
    def broadcast(self, message: Message, device_id: DeviceIDs) -> None: ...
    def _broadcast_worker(self) -> None: ...

    @property
    def _queueids_to_queues(self) -> QueueIdsToQueues: ...

    @property
    def _user_id(self) -> Optional[str]: ...

    @property
    def chain_id(self) -> ChainID: ...

    def _initialize_first_sync(self) -> None: ...
    def _initialize_sync(self) -> None: ...
    def _validate_matrix_messages(
        self, messages: List[MatrixMessage]
    ) -> Tuple[List[ReceivedRaidenMessage], List[ReceivedCallMessage]]: ...
    def _process_raiden_messages(
        self, all_messages: List[ReceivedRaidenMessage]
    ) -> None: ...
    def _handle_messages(self, messages: List[MatrixMessage]) -> bool: ...
    def _process_call_messages(
        self, call_messages: List[ReceivedCallMessage]
    ) -> None: ...
    def _get_retrier(self, receiver: Address) -> _RetryQueue: ...
    def _send_with_retry(self, queue: MessagesQueue) -> None: ...
    def _multicast_services(
        self, data: str, device_id: str = ...
    ) -> None: ...
    def _send_to_device_raw(
        self,
        user_ids: Set[UserID],
        data: str,
        device_id: str = ...,
        message_type: MatrixMessageType = ...,
    ) -> None: ...
    def _send_raw(
        self,
        receiver_address: Address,
        data: str,
        message_type: MatrixMessageType = ...,
        receiver_metadata: Optional[AddressMetadata] = ...,
    ) -> None: ...
    def _send_signaling_message(self, address: Address, message: str) -> None: ...

def _query_metadata(
    pfs_proxy: PFSProxy, address: Address
) -> Optional[AddressMetadata]: ...

def populate_services_addresses(
    transport: MatrixTransport,
    service_registry: Optional[ServiceRegistry],
    block_identifier: BlockIdentifier,
) -> None: ...