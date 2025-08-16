from typing import TYPE_CHECKING, Counter as CounterType, Dict, List, Optional, Set, Tuple, Any, Callable, Iterable, Iterator, NamedTuple, Address, AddressHex, AddressMetadata, UserID

class MessagesQueue:
    pass

class _RetryQueue(Runnable):
    class _MessageData(NamedTuple):
        queue_identifier: QueueIdentifier
        message: Message
        text: str
        expiration_generator: Iterator[bool]
        address_metadata: Optional[AddressMetadata]

    def __init__(self, transport: 'MatrixTransport', receiver: Address):
        self.transport: 'MatrixTransport' = transport
        self.receiver: Address = receiver
        self._message_queue: List[_MessageData] = []
        self._notify_event: Event = gevent.event.Event()
        self._idle_since: int = 0

    def enqueue(self, queue_identifier: QueueIdentifier, messages: List[Tuple[Message, Optional[AddressMetadata]]]):
        ...

    def enqueue_unordered(self, message: Message, address_metadata: Optional[AddressMetadata] = None):
        ...

    def notify(self):
        ...

    def _check_and_send(self):
        ...

    def _expiration_generator(timeout_generator: Callable[[], Iterator[bool]], now: Callable[[], float] = time.time) -> Iterator[bool]:
        ...

    def _run(self):
        ...

    @property
    def is_idle(self) -> bool:
        ...

class MatrixTransport(Runnable):
    def __init__(self, config: MatrixTransportConfig, environment: Environment, enable_tracing: bool = False):
        ...

    def start(self, raiden_service: 'RaidenService', prev_auth_data: Dict[str, Any]):
        ...

    def health_check_web_rtc(self, partner: Address):
        ...

    def _set_presence(self, state: UserPresence):
        ...

    def _run(self):
        ...

    def stop(self):
        ...

    def send_async(self, message_queues: List[MessagesQueue]):
        ...

    def update_services_addresses(self, addresses_validity: Dict[Address, int]):
        ...

    def expire_services_addresses(self, current_timestamp: int, block_number: int):
        ...

    def broadcast(self, message: Message, device_id: DeviceIDs):
        ...

    def _broadcast_worker(self):
        ...

    @property
    def _queueids_to_queues(self) -> Dict[QueueIdentifier, MessagesQueue]:
        ...

    @property
    def _user_id(self) -> Optional[UserID]:
        ...

    @property
    def chain_id(self) -> ChainID:
        ...

    def _initialize_first_sync(self):
        ...

    def _initialize_sync(self):
        ...

    def _validate_matrix_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[ReceivedRaidenMessage], List[ReceivedCallMessage]]:
        ...

    def _process_raiden_messages(self, all_messages: List[ReceivedRaidenMessage]):
        ...

    def _handle_messages(self, messages: List[Dict[str, Any]]) -> bool:
        ...

    def _process_call_messages(self, call_messages: List[ReceivedCallMessage]):
        ...

    def _get_retrier(self, receiver: Address) -> _RetryQueue:
        ...

    def _send_with_retry(self, queue: MessagesQueue):
        ...

    def _multicast_services(self, data: str, device_id: str = '*'):
        ...

    def _send_to_device_raw(self, user_ids: Set[UserID], data: str, device_id: str = '*', message_type: MatrixMessageType = MatrixMessageType.TEXT):
        ...

    def _send_raw(self, receiver_address: Address, data: str, message_type: MatrixMessageType = MatrixMessageType.TEXT, receiver_metadata: Optional[AddressMetadata] = None):
        ...

    def _send_signaling_message(self, address: Address, message: str):
        ...

def _query_metadata(pfs_proxy: PFSProxy, address: Address) -> Optional[AddressMetadata]:
    ...

def populate_services_addresses(transport: MatrixTransport, service_registry: ServiceRegistry, block_identifier: BlockIdentifier):
    ...
