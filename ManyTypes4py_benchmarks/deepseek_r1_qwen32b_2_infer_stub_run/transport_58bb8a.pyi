import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Counter as CounterType,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)
import uuid

from eth_utils import Address
from gevent.event import Event
from gevent.queue import JoinableQueue
from matrix_client.client import GMatrixClient
from raiden.constants import CommunicationMedium, DeviceIDs, Environment, MatrixMessageType
from raiden.exceptions import RaidenUnrecoverableError, TransportError
from raiden.messages.abstract import Message, RetrieableMessage, SignedRetrieableMessage
from raiden.messages.healthcheck import Ping, Pong
from raiden.messages.synchronization import Delivered, Processed
from raiden.network.pathfinding import PFSProxy
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.settings import MatrixTransportConfig
from raiden.storage.serialization.serializer import MessageSerializer
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, QueueIdentifier
from raiden.utils.typing import (
    AddressHex,
    AddressMetadata,
    ChainID,
    DictSerializer,
    MessageID,
    UserID,
)

RETRY_QUEUE_IDLE_AFTER = 10
SET_PRESENCE_INTERVAL = 60


@dataclass
class MessagesQueue:
    pass


def _metadata_key_func(message_data: Any) -> str:
    ...


class _RetryQueue:
    class _MessageData(NamedTuple):
        queue_identifier: QueueIdentifier
        message: Message
        text: str
        expiration_generator: Generator[bool, None, None]
        address_metadata: Optional[AddressMetadata]

    def __init__(self, transport: 'MatrixTransport', receiver: Address):
        ...

    @property
    def log(self) -> structlog.BoundLogger:
        ...

    def enqueue(
        self,
        queue_identifier: QueueIdentifier,
        messages: List[Tuple[Message, Optional[AddressMetadata]]],
    ) -> None:
        ...

    def enqueue_unordered(
        self, message: Message, address_metadata: Optional[AddressMetadata] = None
    ) -> None:
        ...

    def notify(self) -> None:
        ...

    def _check_and_send(self) -> None:
        ...

    def _run(self) -> None:
        ...

    @property
    def is_idle(self) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...


class MatrixTransport:
    log: structlog.BoundLogger

    def __init__(
        self,
        config: MatrixTransportConfig,
        environment: Environment,
        enable_tracing: bool = False,
    ) -> None:
        ...

    @property
    def started(self) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def checksummed_address(self) -> Optional[str]:
        ...

    @property
    def _node_address(self) -> Optional[Address]:
        ...

    @property
    def user_id(self) -> Optional[str]:
        ...

    @property
    def displayname(self) -> str:
        ...

    @property
    def address_metadata(self) -> Optional[AddressMetadata]:
        ...

    def start(self, raiden_service: 'RaidenService', prev_auth_data: Dict[str, Any]) -> None:
        ...

    def health_check_web_rtc(self, partner: Address) -> None:
        ...

    def _set_presence(self, state: UserPresence) -> None:
        ...

    def _run(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def send_async(self, message_queues: List[MessagesQueue]) -> None:
        ...

    def update_services_addresses(self, addresses_validity: Dict[Address, int]) -> None:
        ...

    def expire_services_addresses(
        self, current_timestamp: int, block_number: int
    ) -> None:
        ...

    def broadcast(self, message: Message, device_id: str) -> None:
        ...

    def _broadcast_worker(self) -> None:
        ...

    @property
    def _queueids_to_queues(self) -> QueueIdsToQueues:
        ...

    @property
    def _user_id(self) -> Optional[str]:
        ...

    @property
    def chain_id(self) -> ChainID:
        ...

    def _initialize_first_sync(self) -> None:
        ...

    def _initialize_sync(self) -> None:
        ...

    def _validate_matrix_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[ReceivedRaidenMessage], List[ReceivedCallMessage]]:
        ...

    def _process_raiden_messages(
        self, all_messages: List[ReceivedRaidenMessage]
    ) -> None:
        ...

    def _handle_messages(self, messages: List[Dict[str, Any]]) -> bool:
        ...

    def _process_call_messages(
        self, call_messages: List[ReceivedCallMessage]
    ) -> None:
        ...

    def _get_retrier(self, receiver: Address) -> _RetryQueue:
        ...

    def _send_with_retry(self, queue: MessagesQueue) -> None:
        ...

    def _multicast_services(self, data: str, device_id: str = '*') -> None:
        ...

    def _send_to_device_raw(
        self,
        user_ids: Set[str],
        data: str,
        device_id: str = '*',
        message_type: MatrixMessageType = MatrixMessageType.TEXT,
    ) -> None:
        ...

    def _send_raw(
        self,
        receiver_address: Address,
        data: str,
        message_type: MatrixMessageType = MatrixMessageType.TEXT,
        receiver_metadata: Optional[AddressMetadata] = None,
    ) -> None:
        ...

    def _send_signaling_message(
        self, address: Address, message: str
    ) -> None:
        ...


def _query_metadata(pfs_proxy: PFSProxy, address: Address) -> Optional[AddressMetadata]:
    ...


def populate_services_addresses(
    transport: MatrixTransport,
    service_registry: ServiceRegistry,
    block_identifier: BlockIdentifier,
) -> None:
    ...