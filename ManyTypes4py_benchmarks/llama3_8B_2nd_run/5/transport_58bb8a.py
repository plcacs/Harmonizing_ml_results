import asyncio
import itertools
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from json import JSONDecodeError
from random import randint
from typing import TYPE_CHECKING, Counter as CounterType
from urllib.parse import urlparse
from uuid import uuid4
import gevent
import structlog
from eth_utils import encode_hex, is_binary_address, to_normalized_address
from gevent.event import Event
from gevent.queue import JoinableQueue
from matrix_client.errors import MatrixHttpLibError
from web3.types import BlockIdentifier
from raiden.constants import EMPTY_SIGNATURE, MATRIX_AUTO_SELECT_SERVER, CommunicationMedium, DeviceIDs, Environment, MatrixMessageType
from raiden.exceptions import RaidenUnrecoverableError, TransportError
from raiden.messages.abstract import Message, RetrieableMessage, SignedRetrieableMessage
from raiden.messages.healthcheck import Ping, Pong
from raiden.messages.synchronization import Delivered, Processed
from raiden.network.pathfinding import PFSProxy
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.transport.matrix.client import GMatrixClient, MatrixMessage, ReceivedCallMessage, ReceivedRaidenMessage
from raiden.network.transport.matrix.rtc.web_rtc import WebRTCManager
from raiden.network.transport.matrix.utils import DisplayNameCache, MessageAckTimingKeeper, UserPresence, address_from_userid, capabilities_schema, get_user_id_from_metadata, login, make_client, make_message_batches, make_user_id, validate_and_parse_message, validate_userid_signature
from raiden.settings import MatrixTransportConfig
from raiden.storage.serialization import DictSerializer
from raiden.transfer import views
from raiden.transfer.state import QueueIdsToQueues
from raiden.utils.capabilities import capconfig_to_dict
from raiden.utils.formatting import to_checksum_address
from raiden.utils.logging import redact_secret
from raiden.utils.runnable import Runnable
from raiden.utils.system import get_system_spec
from raiden.utils.tracing import matrix_client_enable_requests_tracing
from typing import Optional, Set, Tuple, List, Dict, Any, Callable, MYPY_ANNOTATION

class MessagesQueue(Runnable):
    ...

class _RetryQueue(Runnable):
    ...

class MatrixTransport(Runnable):
    ...

    def start(self, raiden_service: 'RaidenService', prev_auth_data: Any) -> None:
        ...

    def send_async(self, message_queues: List[MessagesQueue]) -> None:
        ...

    def update_services_addresses(self, addresses_validity: Dict[Address, int]) -> None:
        ...

    def expire_services_addresses(self, current_timestamp: int, block_number: int) -> None:
        ...

    def broadcast(self, message: Message, device_id: DeviceIDs) -> None:
        ...

    def _send_with_retry(self, queue: MessagesQueue) -> None:
        ...

    def _multicast_services(self, data: Any, device_id: str = '*') -> None:
        ...

    def _send_raw(self, receiver_address: Address, data: Any, message_type: MatrixMessageType = MatrixMessageType.TEXT, receiver_metadata: Optional[Dict] = None) -> None:
        ...

    def _send_signaling_message(self, address: Address, message: Any) -> None:
        ...

    def _process_raiden_messages(self, all_messages: List[ReceivedRaidenMessage]) -> None:
        ...

    def _handle_messages(self, messages: List[MatrixMessage]) -> bool:
        ...

    def _process_call_messages(self, call_messages: List[ReceivedCallMessage]) -> None:
        ...

    def _get_retrier(self, receiver: Address) -> _RetryQueue:
        ...

    def _send_to_device_raw(self, user_ids: Set[UserID], data: Any, device_id: str = '*', message_type: MatrixMessageType = MatrixMessageType.TEXT) -> None:
        ...

    def _populate_services_addresses(self, transport: 'MatrixTransport', service_registry: ServiceRegistry, block_identifier: BlockIdentifier) -> None:
        ...
