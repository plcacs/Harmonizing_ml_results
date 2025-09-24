import random
from unittest.mock import MagicMock, Mock
import gevent
import pytest
from eth_utils import to_normalized_address
from gevent import Timeout
import raiden
from raiden.constants import BLOCK_ID_LATEST, EMPTY_SIGNATURE, DeviceIDs, Environment, RoutingMode
from raiden.messages.monitoring_service import RequestMonitoring
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.messages.synchronization import Delivered, Processed
from raiden.network.transport.matrix.transport import MatrixTransport, MessagesQueue, _RetryQueue, populate_services_addresses
from raiden.network.transport.matrix.utils import AddressReachability
from raiden.services import send_pfs_update, update_monitoring_service_from_balance_proof
from raiden.settings import MIN_MONITORING_AMOUNT_DAI, MONITORING_REWARD, CapabilitiesConfig, MatrixTransportConfig, RaidenConfig, ServiceConfig
from raiden.storage.serialization.serializer import MessageSerializer
from raiden.tests.utils import factories
from raiden.tests.utils.factories import HOP1, CanonicalIdentifierProperties, NettingChannelEndStateProperties, make_privkeys_ordered
from raiden.tests.utils.mocks import MockRaidenService, make_pfs_config
from raiden.tests.utils.smartcontracts import deploy_service_registry_and_set_urls
from raiden.transfer import views
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, QueueIdentifier
from raiden.utils.keys import privatekey_to_address
from raiden.utils.typing import Address, MessageID
from raiden_contracts.utils.type_aliases import ChainID
from typing import Set, List, Tuple, Dict, Any, Optional, Union, Callable
from web3 import Web3
from web3.contract import Contract
from eth_typing import ChecksumAddress

HOP1_BALANCE_PROOF = factories.BalanceProofSignedStateProperties(pkey=factories.HOP1_KEY)
TIMEOUT_MESSAGE_RECEIVE: int = 15

@pytest.fixture
def num_services() -> int:
    return 2

@pytest.fixture
def services(num_services: int, matrix_transports: List[MatrixTransport]) -> List[str]:
    service_addresses_to_expiry: Dict[Address, int] = {factories.make_address(): 9999 for _ in range(num_services)}
    for transport in matrix_transports:
        transport.update_services_addresses(service_addresses_to_expiry)
    return [to_normalized_address(addr) for addr in service_addresses_to_expiry.keys()]

@pytest.fixture
def number_of_transports() -> int:
    return 1

class MessageHandler:

    def __init__(self, bag: Set[Any]) -> None:
        self.bag = bag

    def on_messages(self, _: Any, messages: List[Any]) -> None:
        self.bag.update(messages)

def get_to_device_broadcast_messages(to_device_mock: MagicMock, expected_receiver_addresses: List[str], device_id: DeviceIDs) -> List[Any]:
    collected_messages: List[Any] = []
    for (_, kwargs) in to_device_mock.call_args_list:
        assert kwargs['event_type'] == 'm.room.message'
        messages_batch: List[Any] = []
        addresses: List[str] = []
        for (address, to_device_dict) in kwargs['messages'].items():
            addresses.append(address.split(':')[0][1:])
            assert to_device_dict.keys() == {device_id}
            messages: List[str] = to_device_dict[device_id]['body'].split('\n')
            messages = [MessageSerializer.deserialize(message) for message in messages]
            if not messages_batch:
                messages_batch = messages
            else:
                assert messages_batch == messages
        assert len(addresses) == len(expected_receiver_addresses)
        assert set(addresses) == set(expected_receiver_addresses)
        collected_messages += messages_batch
    return collected_messages

def ping_pong_message_success(transport0: MatrixTransport, transport1: MatrixTransport) -> bool:
    queueid0: QueueIdentifier = QueueIdentifier(recipient=transport0._raiden_service.address, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
    queueid1: QueueIdentifier = QueueIdentifier(recipient=transport1._raiden_service.address, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
    transport0_raiden_queues: Dict[QueueIdentifier, List[Any]] = views.get_all_messagequeues(views.state_from_raiden(transport0._raiden_service))
    transport1_raiden_queues: Dict[QueueIdentifier, List[Any]] = views.get_all_messagequeues(views.state_from_raiden(transport1._raiden_service))
    transport0_raiden_queues[queueid1] = []
    transport1_raiden_queues[queueid0] = []
    received_messages0: Set[Any] = transport0._raiden_service.message_handler.bag
    received_messages1: Set[Any] = transport1._raiden_service.message_handler.bag
    msg_id: float = random.randint(100000.0, 900000.0)
    ping_message: Processed = Processed(message_identifier=MessageID(msg_id), signature=EMPTY_SIGNATURE)
    pong_message: Delivered = Delivered(delivered_message_identifier=MessageID(msg_id), signature=EMPTY_SIGNATURE)
    transport0_raiden_queues[queueid1].append(ping_message)
    transport0._raiden_service.sign(ping_message)
    transport1._raiden_service.sign(pong_message)
    transport0.send_async([MessagesQueue(queueid1, [(ping_message, transport1.address_metadata)])])
    with Timeout(TIMEOUT_MESSAGE_RECEIVE, exception=False):
        all_messages_received: bool = False
        while not all_messages_received:
            all_messages_received = ping_message in received_messages1 and pong_message in received_messages0
            gevent.sleep(0.1)
    assert ping_message in received_messages1
    assert pong_message in received_messages0
    transport0_raiden_queues[queueid1].clear()
    transport1_raiden_queues[queueid0].append(ping_message)
    transport0._raiden_service.sign(pong_message)
    transport1._raiden_service.sign(ping_message)
    transport1.send_async([MessagesQueue(queueid0, [(ping_message, transport0.address_metadata)])])
    with Timeout(TIMEOUT_MESSAGE_RECEIVE, exception=False):
        all_messages_received = False
        while not all_messages_received:
            all_messages_received = ping_message in received_messages0 and pong_message in received_messages1
            gevent.sleep(0.1)
    assert ping_message in received_messages0
    assert pong_message in received_messages1
    transport1_raiden_queues[queueid0].clear()
    return all_messages_received

def is_reachable(transport: MatrixTransport, address: Address) -> bool:
    raise NotImplementedError

def _wait_for_peer_reachability(transport: MatrixTransport, target_address: Address, target_reachability: AddressReachability, timeout: int = 5) -> None:
    raise NotImplementedError
    with Timeout(timeout):
        while True:
            gevent.sleep(0.1)

def wait_for_peer_unreachable(transport: MatrixTransport, target_address: Address, timeout: int = 5) -> None:
    _wait_for_peer_reachability(transport=transport, target_address=target_address, target_reachability=AddressReachability.UNREACHABLE, timeout=timeout)

def wait_for_peer_reachable(transport: MatrixTransport, target_address: Address, timeout: int = 5) -> None:
    _wait_for_peer_reachability(transport=transport, target_address=target_address, target_reachability=AddressReachability.REACHABLE, timeout=timeout)

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_sync(matrix_transports: List[MatrixTransport]) -> None:
    (transport0, transport1) = matrix_transports
    transport0_messages: Set[Any] = set()
    transport1_messages: Set[Any] = set()
    transport0_message_handler: MessageHandler = MessageHandler(transport0_messages)
    transport1_message_handler: MessageHandler = MessageHandler(transport1_messages)
    raiden_service0: MockRaidenService = MockRaidenService(transport0_message_handler)
    raiden_service1: MockRaidenService = MockRaidenService(transport1_message_handler)
    raiden_service1.handle_and_track_state_changes = MagicMock()
    transport0.start(raiden_service0, None)
    transport1.start(raiden_service1, None)
    queue_identifier: QueueIdentifier = QueueIdentifier(recipient=transport1._raiden_service.address, canonical_identifier=factories.UNIT_CANONICAL_ID)
    raiden0_queues: Dict[QueueIdentifier, List[Any]] = views.get_all_messagequeues(views.state_from_raiden(raiden_service0))
    raiden0_queues[queue_identifier] = []
    for i in range(5):
        message: Processed = Processed(message_identifier=MessageID(i), signature=EMPTY_SIGNATURE)
        raiden0_queues[queue_identifier].append(message)
        transport0._raiden_service.sign(message)
        transport0.send_async([MessagesQueue(queue_identifier, [(message, transport1.address_metadata)])])
    with Timeout(TIMEOUT_MESSAGE_RECEIVE):
        while not len(transport0_messages) == 5:
            gevent.sleep(0.1)
        while not len(transport1_messages) == 5:
            gevent.sleep(0.1)
    for i in range(5):
        assert any((m.message_identifier == i for m in transport1_messages))
    for i in range(5):
        assert any((m.delivered_message_identifier == i for m in transport0_messages))
    raiden0_queues[queue_identifier] = []
    transport1.stop()
    for i in range(10, 15):
        message = Processed(message_identifier=MessageID(i), signature=EMPTY_SIGNATURE)
        raiden0_queues[queue_identifier].append(message)
        transport0._raiden_service.sign(message)
        transport0.send_async([MessagesQueue(queue_identifier, [(message, transport1.address_metadata)])])
    transport1.start(transport1._raiden_service, None)
    with gevent.Timeout(TIMEOUT_MESSAGE_RECEIVE):
        while len(transport1_messages) != 10:
            gevent.sleep(0.1)
        while len(transport0_messages) != 10:
            gevent.sleep(0.1)
    for i in range(10, 15):
        assert any((m.message_identifier == i for m in transport1_messages))
    for i in range(10, 15):
        assert any((m.delivered_message_identifier == i for m in transport0_messages))

def test_matrix_message_retry(local_matrix_servers: List[str], retry_interval_initial: float, retry_interval_max: float, retries_before_backoff: int) -> None:
    """Test the retry mechanism implemented into the matrix client.
    The test creates a transport and sends a message. The receiver
    doesn't respond in time. The retrier should then
    wait for the `retry_interval` duration to pass and send the message
    again.
    """
    partner_address: Address = factories.make_address()
    transport: MatrixTransport = MatrixTransport(config=MatrixTransportConfig(retries_before_backoff=retries_before_backoff, retry_interval_initial=retry_interval_initial, retry_interval_max=retry_interval_max, server=local_matrix_servers[0], available_servers=[local_matrix_servers[0]]), environment=Environment.DEVELOPMENT)
    transport._send_raw = MagicMock()
    raiden_service: MockRaidenService = MockRaidenService(None)
    transport.start(raiden_service, None)
    transport.log = MagicMock()
    queueid: QueueIdentifier = QueueIdentifier(recipient=partner_address, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
    chain_state: Any = raiden_service.wal.get_current_state()
    retry_queue: _RetryQueue = transport._get_retrier(partner_address)
    assert bool(retry_queue), 'retry_queue not running'
    message: Processed = Processed(message_identifier=MessageID(0), signature=EMPTY_SIGNATURE)
    transport._raiden_service.sign(message)
    chain_state.queueids_to_queues[queueid] = [message]
    retry_queue.enqueue_unordered(message)
    gevent.idle()
    assert transport._send_raw.call_count == 1
    with gevent.Timeout(retry_interval_initial + 2):
        while transport._send_raw.call_count != 2:
            gevent.sleep(0.1)
    transport.stop()
    transport.greenlet.get()

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_transport_handles_metadata(matrix_transports: List[MatrixTransport]) -> None:
    (transport0, transport1) = matrix_transports
    transport0_messages: Set[Any] = set()
    transport1_messages: Set[Any] = set()
    transport0_message_handler: MessageHandler = MessageHandler(transport0_messages)
    transport1_message_handler: MessageHandler = MessageHandler(transport1_messages)
    raiden_service0: MockRaidenService = MockRaidenService(transport0_message_handler)
    raiden_service1: MockRaidenService = MockRaidenService(transport1_message_handler)
    raiden_service1.handle_and_track_state_changes = MagicMock()
    transport0.start(raiden_service0, None)
    transport1.start(raiden_service1, None)
    queue_identifier: QueueIdentifier = QueueIdentifier(recipient=transport1._raiden_service.address, canonical_identifier=factories.UNIT_CANONICAL_ID)
    raiden0_queues: Dict[QueueIdentifier, List[Any]] = views.get_all_messagequeues(views.state_from_raiden(raiden_service0))
    raiden0_queues[queue_identifier] = []
    correct_metadata: Dict[str, str] = {'user_id': transport1.user_id}
    invalid_metadata: Dict[str, str] = {'user_id': 'invalid'}
    no_metadata: None = None
    all_metadata: Tuple[Optional[Dict[str, str]], ...] = (correct_metadata, invalid_metadata, no_metadata)
    num_sends: int = 2
    message_id: int = 0
    for metadata in all_metadata:
        for _ in range(num_sends):
            message: Processed = Processed(message_identifier=message_id, signature=EMPTY_SIGNATURE)
            raiden0_queues[queue_identifier].append(message)
            transport0._raiden_service.sign(message)
            message_queues: List[MessagesQueue] = [MessagesQueue(queue_identifier, [(message, metadata)])]
            transport0.send_async(message_queues)
            message_id += 1
    num_expected_messages: int = num_sends
    with Timeout(TIMEOUT_MESSAGE_RECEIVE):
        while len(transport0_messages) < num_expected_messages:
            gevent.sleep(0.1)
        while len(transport1_messages) < num_expected_messages:
            gevent.sleep(0.1)
    ids: List[int] = [m.message_identifier for m in transport1_messages]
    delivered_ids: List[int] = [m.delivered_message_identifier for m in transport0_messages]
    assert sorted(ids) == sorted(delivered_ids)
    assert sorted(ids) == list(range(0, num_sends))
    assert len(transport0_messages) == num_expected_messages
    assert len(transport1_messages) == num_expected_messages
    transport0.stop()
    transport1.stop()

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [3])
def test_matrix_cross_server_with_load_balance(matrix_transports: List[MatrixTransport]) -> None:
    (transport0, transport1, transport2) = matrix_transports
    received_messages0: Set[Any] = set()
    received_messages1: Set[Any] = set()
    received_messages2: Set[Any] = set()
    message_handler0: MessageHandler = MessageHandler(received_messages0)
    message_handler1: MessageHandler = MessageHandler(received_messages1)
    message_handler2: MessageHandler = MessageHandler(received_messages2)
    raiden_service0: MockRaidenService = MockRaidenService(message_handler0)
    raiden_service1: MockRaidenService = MockRaidenService(message_handler1)
    raiden_service2: MockRaidenService = MockRaidenService(message_handler2)
    transport0.start(raiden_service0, '')
    transport1.start(raiden_service1, '')
    transport2.start(raiden_service2, '')
    assert ping_pong_message_success(transport0, transport1)
    assert ping_pong_message_success(transport0, transport2)
    assert ping_pong_message_success(transport1, transport0)
    assert ping_pong_message_success(transport1, transport2)
    assert ping_pong_message_success(transport2, transport0)
    assert ping_pong_message_success(transport2, transport1)

@pytest.mark.parametrize('device_id', (DeviceIDs.PFS, DeviceIDs.MS))
def test_matrix_broadcast(matrix_transports: List[MatrixTransport], services: List[str], device_id: DeviceIDs) -> None:
   