import asyncio
import itertools
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from json import JSONDecodeError
from random import randint
from typing import TYPE_CHECKING, Counter as CounterType, Dict, List, Optional, Set, Tuple, Any, Iterable, Iterator, Callable, NamedTuple, cast
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
from raiden.network.transport.utils import timeout_exponential_backoff
from raiden.settings import MatrixTransportConfig
from raiden.storage.serialization import DictSerializer
from raiden.storage.serialization.serializer import MessageSerializer
from raiden.transfer import views
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, QueueIdentifier
from raiden.transfer.state import QueueIdsToQueues
from raiden.utils.capabilities import capconfig_to_dict
from raiden.utils.formatting import to_checksum_address
from raiden.utils.logging import redact_secret
from raiden.utils.runnable import Runnable
from raiden.utils.system import get_system_spec
from raiden.utils.tracing import matrix_client_enable_requests_tracing
from raiden.utils.typing import MYPY_ANNOTATION, Address, AddressHex, AddressMetadata, ChainID, MessageID, UserID

if TYPE_CHECKING:
    from raiden.raiden_service import RaidenService

log = structlog.get_logger(__name__)
RETRY_QUEUE_IDLE_AFTER = 10
SET_PRESENCE_INTERVAL = 60

@dataclass
class MessagesQueue:
    queue_identifier: QueueIdentifier
    messages: List[Tuple[Message, Optional[AddressMetadata]]]

def _metadata_key_func(message_data: Any) -> str:
    address_metadata = message_data.address_metadata
    if address_metadata is None:
        return ''
    uid = address_metadata.get('user_id', '')
    return uid

class _RetryQueue(Runnable):
    """A helper Runnable to send batched messages to receiver through transport"""

    class _MessageData(NamedTuple):
        """Small helper data structure for message queue"""
        queue_identifier: QueueIdentifier
        message: Message
        text: str
        expiration_generator: Iterator[bool]
        address_metadata: Optional[AddressMetadata]

    def __init__(self, transport: 'MatrixTransport', receiver: Address) -> None:
        self.transport = transport
        self.receiver = receiver
        self._message_queue: List['_RetryQueue._MessageData'] = []
        self._notify_event = gevent.event.Event()
        self._idle_since: int = 0
        super().__init__()
        self.greenlet.name = f'RetryQueue recipient:{to_checksum_address(self.receiver)}'

    @property
    def log(self) -> Any:
        return self.transport.log

    @staticmethod
    def _expiration_generator(timeout_generator: Iterable[float], now: Callable[[], float] = time.time) -> Iterator[bool]:
        """Stateful generator that yields True if more than timeout has passed since previous True,
        False otherwise.

        Helper method to tell when a message needs to be retried (more than timeout seconds
        passed since last time it was sent).
        timeout is iteratively fetched from timeout_generator
        First value is True to always send message at least once
        """
        for timeout in timeout_generator:
            _next = now() + timeout
            yield True
            while now() < _next:
                yield False

    def enqueue(self, queue_identifier: QueueIdentifier, messages: List[Tuple[Message, Optional[AddressMetadata]]]) -> None:
        """Enqueue a message to be sent, and notify main loop"""
        msg = f'queue_identifier.recipient ({to_checksum_address(queue_identifier.recipient)})  must match self.receiver ({to_checksum_address(self.receiver)}).'
        assert queue_identifier.recipient == self.receiver, msg
        timeout_generator = timeout_exponential_backoff(self.transport._config.retries_before_backoff, self.transport._config.retry_interval_initial, self.transport._config.retry_interval_max)
        encoded_messages: List['_RetryQueue._MessageData'] = []
        for (message, address_metadata) in messages:
            already_queued = any((queue_identifier == data.queue_identifier and message == data.message for data in self._message_queue))
            if already_queued:
                self.log.warning('Message already in queue - ignoring', receiver=to_checksum_address(self.receiver), queue=queue_identifier, message=redact_secret(DictSerializer.serialize(message)))
            else:
                expiration_generator = self._expiration_generator(timeout_generator)
                data = _RetryQueue._MessageData(queue_identifier=queue_identifier, message=message, text=MessageSerializer.serialize(message), expiration_generator=expiration_generator, address_metadata=address_metadata)
                encoded_messages.append(data)
        self._message_queue.extend(encoded_messages)
        self.notify()

    def enqueue_unordered(self, message: Message, address_metadata: Optional[AddressMetadata] = None) -> None:
        """Helper to enqueue a message in the unordered queue."""
        self.enqueue(queue_identifier=QueueIdentifier(recipient=self.receiver, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE), messages=[(message, address_metadata)])

    def notify(self) -> None:
        """Notify main loop to check if anything needs to be sent"""
        self._notify_event.set()

    def _check_and_send(self) -> None:
        """Check and send all pending/queued messages that are not waiting on retry timeout

        After composing the to-be-sent message, also message queue from messages that are not
        present in the respective SendMessageEvent queue anymore
        """
        if not self.transport.greenlet:
            self.log.warning("Can't retry", reason='Transport not yet started')
            return
        if self.transport._stop_event.ready():
            self.log.warning("Can't retry", reason='Transport stopped')
            return
        if self.transport._prioritize_broadcast_messages:
            self.transport._broadcast_queue.join()
        self.log.debug('Retrying message(s)', receiver=to_checksum_address(self.receiver), queue_size=len(self._message_queue))

        def message_is_in_queue(message_data: '_RetryQueue._MessageData') -> bool:
            if message_data.queue_identifier not in self.transport._queueids_to_queues:
                return False
            return any((isinstance(message_data.message, RetrieableMessage) and send_event.message_identifier == message_data.message.message_identifier for send_event in self.transport._queueids_to_queues[message_data.queue_identifier]))
        queue_by_user_id = sorted(self._message_queue[:], key=_metadata_key_func)
        for (user_id, batch) in itertools.groupby(queue_by_user_id, _metadata_key_func):
            message_data_batch = list(batch)
            if user_id == '':
                address_metadata = None
            else:
                address_metadata = message_data_batch[0].address_metadata
            message_texts: List[str] = []
            for message_data in message_data_batch:
                remove = False
                if isinstance(message_data.message, (Delivered, Ping, Pong)):
                    remove = True
                    message_texts.append(message_data.text)
                elif not message_is_in_queue(message_data):
                    remove = True
                    self.log.debug('Stopping message send retry', queue=message_data.queue_identifier, message=message_data.message, reason='Message was removed from queue or queue was removed')
                elif next(message_data.expiration_generator):
                    message_texts.append(message_data.text)
                    if self.transport._environment is Environment.DEVELOPMENT:
                        if isinstance(message_data.message, RetrieableMessage):
                            self.transport._counters['retry'][message_data.message.__class__.__name__, message_data.message.message_identifier] += 1
                if remove:
                    self._message_queue.remove(message_data)
            if message_texts:
                self.log.debug('Send', receiver=to_checksum_address(self.receiver), messages=message_texts)
                for message_batch in make_message_batches(message_texts):
                    self.transport._send_raw(self.receiver, message_batch, receiver_metadata=address_metadata)

    def _run(self) -> None:
        msg = '_RetryQueue started before transport._raiden_service is set. _RetryQueue should not be started before transport.start() is called'
        assert self.transport._raiden_service is not None, msg
        self.greenlet.name = f'RetryQueue node:{to_checksum_address(self.transport._raiden_service.address)} recipient:{to_checksum_address(self.receiver)}'
        while not self.transport._stop_event.ready():
            self._notify_event.clear()
            if self._message_queue:
                self._idle_since = 0
                self._check_and_send()
            else:
                self._idle_since += 1
            if self.is_idle:
                self.log.debug('Exiting idle RetryQueue', queue=self)
                return
            self._notify_event.wait(self.transport._config.retry_interval_initial)

    @property
    def is_idle(self) -> bool:
        return self._idle_since >= RETRY_QUEUE_IDLE_AFTER

    def __str__(self) -> str:
        return self.greenlet.name

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} for {to_normalized_address(self.receiver)}>'

class MatrixTransport(Runnable):
    log = log

    def __init__(self, config: MatrixTransportConfig, environment: Environment, enable_tracing: Optional[bool] = False) -> None:
        super().__init__()
        self._uuid = uuid4()
        self._config = config
        self._environment = environment
        self._raiden_service: Optional['RaidenService'] = None
        self._web_rtc_manager: Optional[WebRTCManager] = None
        if config.server == MATRIX_AUTO_SELECT_SERVER:
            homeserver_candidates = config.available_servers
        elif urlparse(config.server).scheme in {'http', 'https'}:
            homeserver_candidates = [config.server]
        else:
            raise TransportError(f"Invalid matrix server specified (valid values: '{MATRIX_AUTO_SELECT_SERVER}' or a URL)")

        def _http_retry_delay() -> Iterable[float]:
            return timeout_exponential_backoff(self._config.retries_before_backoff, self._config.retry_interval_initial, self._config.retry_interval_max)
        version = get_system_spec()['raiden']
        self._client: GMatrixClient = make_client(self._handle_messages, homeserver_candidates, http_pool_maxsize=4, http_retry_timeout=40, http_retry_delay=_http_retry_delay, environment=environment, user_agent=f'Raiden {version}')
        if enable_tracing:
            matrix_client_enable_requests_tracing(self._client)
        self.server_url = self._client.api.base_url
        self._server_name = urlparse(self.server_url).netloc
        self._all_server_names = {self._server_name}
        for server_url in config.available_servers:
            self._all_server_names.add(urlparse(server_url).netloc)
        msg = 'There needs to be at least one matrix server known.'
        assert self._all_server_names, msg
        self.greenlets: List[gevent.Greenlet] = []
        self._address_to_retrier: Dict[Address, _RetryQueue] = {}
        self._displayname_cache = DisplayNameCache()
        self._broadcast_queue: JoinableQueue[Tuple[str, Message]] = JoinableQueue()
        self._started = False
        self._stop_event: Event = Event()
        self._stop_event.set()
        self._broadcast_event = Event()
        self._prioritize_broadcast_messages: bool = True
        self._counters: Dict[str, CounterType[Tuple[str, MessageID]]] = {}
        self._message_timing_keeper: Optional[MessageAckTimingKeeper] = None
        if environment is Environment.DEVELOPMENT:
            self._counters['send'] = Counter()
            self._counters['retry'] = Counter()
            self._counters['dispatch'] = Counter()
            self._message_timing_keeper = MessageAckTimingKeeper()
        self.services_addresses: Dict[Address, int] = {}

    @property
    def started(self) -> bool:
        return self._started

    def __repr__(self) -> str:
        if self._raiden_service is not None:
            node = f' node:{self.checksummed_address}'
        else:
            node = ''
        return f'<{self.__class__.__name__}{node} id:{self._uuid}>'

    @property
    def checksummed_address(self) -> Optional[AddressHex]:
        assert self._raiden_service is not None, '_raiden_service not set'
        address = self._node_address
        if address is None:
            return None
        return to_checksum_address(self._raiden_service.address)

    @property
    def _node_address(self) -> Optional[Address]:
        return self._raiden_service.address if self._raiden_service is not None else None

    @property
    def user_id(self) -> Optional[UserID]:
        address = self._node_address
        return make_user_id(address, self._server_name) if address is not None else None

    @property
    def displayname(self) -> str:
        if self._raiden_service is None:
            return ''
        signature_bytes = self._raiden_service.signer.sign(str(self.user_id).encode())
        return encode_hex(signature_bytes)

    @property
    def address_metadata(self) -> Optional[AddressMetadata]:
        cap_dict = capconfig_to_dict(self._config.capabilities_config)
        own_caps = capabilities_schema.dump({'capabilities': cap_dict})['capabilities']
        own_user_id = self.user_id
        if own_user_id is None:
            return None
        return dict(user_id=own_user_id, capabilities=own_caps, displayname=self.displayname)

    def start(self, raiden_service: 'RaidenService', prev_auth_data: Optional[str]) -> None:
        if not self._stop_event.ready():
            raise RuntimeError(f'{self!r} already started')
        self.log.debug('Matrix starting')
        self._stop_event.clear()
        self._raiden_service = raiden_service
        assert raiden_service.pfs_proxy is not None, 'must be set'
        self._web_rtc_manager = WebRTCManager(raiden_service.address, self._process_raiden_messages, self._send_signaling_message, self._stop_event)
        self._web_rtc_manager.greenlet.link_exception(self.on_error)
        assert asyncio.get_event_loop().is_running(), 'the loop must be running'
        self.log.debug('Asyncio loop is running', running=asyncio.get_event_loop().is_running())
        try:
            capabilities = capconfig_to_dict(self._config.capabilities_config)
            login(client=self._client, signer=self._raiden_service.signer, device_id=DeviceIDs.RAIDEN, prev_auth_data=prev_auth_data, capabilities=capabilities)
        except ValueError as ex:
            raise RaidenUnrecoverableError('Matrix SDK failed to properly set the userid') from ex
        except MatrixHttpLibError as ex:
            raise RaidenUnrecoverableError('The Matrix homeserver seems to be unavailable.') from ex
        self.log = log.bind(current_user=self._user_id, node=to_checksum_address(self._raiden_service.address), transport_uuid=str(self._uuid))
        self._initialize_first_sync()
        self._initialize_sync()
        for retrier in self._address_to_retrier.values():
            if not retrier:
                self.log.debug('Starting retrier', retrier=retrier)
                retrier.start()
        super().start()
        self._started = True
        self.log.debug('Matrix started', config=self._config)
        self._schedule_new_greenlet(self._set_presence, UserPresence.ONLINE)
        chain_state = views.state_from_raiden(raiden_service)
        for neighbour in views.all_neighbour_nodes(chain_state):
            self.health_check_web_rtc(neighbour)

    def health_check_web_rtc(self, partner: Address) -> None:
        assert self._web_rtc_manager is not None, 'must be set'
        if self._started and (not self._web_rtc_manager.has_ready_channel(partner)):
           