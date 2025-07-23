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
from raiden.network.transport.matrix.transport import (
    MatrixTransport,
    MessagesQueue,
    _RetryQueue,
    populate_services_addresses,
)
from raiden.network.transport.matrix.utils import AddressReachability
from raiden.services import (
    send_pfs_update,
    update_monitoring_service_from_balance_proof,
)
from raiden.settings import (
    MIN_MONITORING_AMOUNT_DAI,
    MONITORING_REWARD,
    CapabilitiesConfig,
    MatrixTransportConfig,
    RaidenConfig,
    ServiceConfig,
)
from raiden.storage.serialization.serializer import MessageSerializer
from raiden.tests.utils import factories
from raiden.tests.utils.factories import (
    HOP1,
    CanonicalIdentifierProperties,
    NettingChannelEndStateProperties,
    make_privkeys_ordered,
)
from raiden.tests.utils.mocks import MockRaidenService, make_pfs_config
from raiden.tests.utils.smartcontracts import deploy_service_registry_and_set_urls
from raiden.transfer import views
from raiden.transfer.identifiers import (
    CANONICAL_IDENTIFIER_UNORDERED_QUEUE,
    QueueIdentifier,
)
from raiden.utils.keys import privatekey_to_address
from raiden.utils.typing import Address, MessageID
from raiden_contracts.utils.type_aliases import ChainID
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

HOP1_BALANCE_PROOF = factories.BalanceProofSignedStateProperties(pkey=factories.HOP1_KEY)
TIMEOUT_MESSAGE_RECEIVE = 15


@pytest.fixture
def num_services() -> int:
    return 2


@pytest.fixture
def services(
    num_services: int, matrix_transports: List[MatrixTransport]
) -> List[str]:
    service_addresses_to_expiry: Dict[str, int] = {
        factories.make_address(): 9999 for _ in range(num_services)
    }
    for transport in matrix_transports:
        transport.update_services_addresses(service_addresses_to_expiry)
    return [to_normalized_address(addr) for addr in service_addresses_to_expiry.keys()]


@pytest.fixture
def number_of_transports() -> int:
    return 1


class MessageHandler:
    def __init__(self, bag: Set[Union[Processed, Delivered]]) -> None:
        self.bag: Set[Union[Processed, Delivered]] = bag

    def on_messages(
        self, _: Any, messages: List[Union[Processed, Delivered]]
    ) -> None:
        self.bag.update(messages)


def get_to_device_broadcast_messages(
    to_device_mock: MagicMock,
    expected_receiver_addresses: List[str],
    device_id: str,
) -> List[Union[Processed, Delivered]]:
    collected_messages: List[Union[Processed, Delivered]] = []
    for _, kwargs in to_device_mock.call_args_list:
        assert kwargs["event_type"] == "m.room.message"
        messages_batch: List[Union[Processed, Delivered]] = []
        addresses: List[str] = []
        for address, to_device_dict in kwargs["messages"].items():
            addresses.append(address.split(":")[0][1:])
            assert set(to_device_dict.keys()) == {device_id}
            messages_str = to_device_dict[device_id]["body"].split("\n")
            messages = [MessageSerializer.deserialize(message) for message in messages_str]
            if not messages_batch:
                messages_batch = messages
            else:
                assert messages_batch == messages
        assert len(addresses) == len(expected_receiver_addresses)
        assert set(addresses) == set(expected_receiver_addresses)
        collected_messages += messages_batch
    return collected_messages


def ping_pong_message_success(
    transport0: MatrixTransport, transport1: MatrixTransport
) -> bool:
    queueid0 = QueueIdentifier(
        recipient=transport0._raiden_service.address,
        canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE,
    )
    queueid1 = QueueIdentifier(
        recipient=transport1._raiden_service.address,
        canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE,
    )
    transport0_raiden_queues: Dict[QueueIdentifier, List[Union[Processed, Delivered]]] = (
        views.get_all_messagequeues(views.state_from_raiden(transport0._raiden_service))
    )
    transport1_raiden_queues: Dict[QueueIdentifier, List[Union[Processed, Delivered]]] = (
        views.get_all_messagequeues(views.state_from_raiden(transport1._raiden_service))
    )
    transport0_raiden_queues[queueid1] = []
    transport1_raiden_queues[queueid0] = []
    received_messages0: Set[Union[Processed, Delivered]] = transport0._raiden_service.message_handler.bag
    received_messages1: Set[Union[Processed, Delivered]] = transport1._raiden_service.message_handler.bag
    msg_id: int = random.randint(100000, 900000)
    ping_message = Processed(message_identifier=MessageID(msg_id), signature=EMPTY_SIGNATURE)
    pong_message = Delivered(
        delivered_message_identifier=MessageID(msg_id), signature=EMPTY_SIGNATURE
    )
    transport0_raiden_queues[queueid1].append(ping_message)
    transport0._raiden_service.sign(ping_message)
    transport1._raiden_service.sign(pong_message)
    transport0.send_async(
        [MessagesQueue(queueid1, [(ping_message, transport1.address_metadata)])]
    )
    with Timeout(TIMEOUT_MESSAGE_RECEIVE, False) as timeout:
        all_messages_received = False
        while not all_messages_received and not timeout.expired:
            all_messages_received = (
                ping_message in received_messages1 and pong_message in received_messages0
            )
            gevent.sleep(0.1)
    assert ping_message in received_messages1
    assert pong_message in received_messages0
    transport0_raiden_queues[queueid1].clear()
    transport1_raiden_queues[queueid0].append(ping_message)
    transport0._raiden_service.sign(pong_message)
    transport1._raiden_service.sign(ping_message)
    transport1.send_async(
        [MessagesQueue(queueid0, [(ping_message, transport0.address_metadata)])]
    )
    with Timeout(TIMEOUT_MESSAGE_RECEIVE, False) as timeout:
        all_messages_received = False
        while not all_messages_received and not timeout.expired:
            all_messages_received = (
                ping_message in received_messages0 and pong_message in received_messages1
            )
            gevent.sleep(0.1)
    assert ping_message in received_messages0
    assert pong_message in received_messages1
    transport1_raiden_queues[queueid0].clear()
    return all_messages_received


def is_reachable(transport: MatrixTransport, address: Address) -> bool:
    raise NotImplementedError


def _wait_for_peer_reachability(
    transport: MatrixTransport,
    target_address: Address,
    target_reachability: AddressReachability,
    timeout: float = 5.0,
) -> None:
    raise NotImplementedError
    with Timeout(timeout):
        while True:
            gevent.sleep(0.1)


def wait_for_peer_unreachable(
    transport: MatrixTransport, target_address: Address, timeout: float = 5.0
) -> None:
    _wait_for_peer_reachability(
        transport=transport,
        target_address=target_address,
        target_reachability=AddressReachability.UNREACHABLE,
        timeout=timeout,
    )


def wait_for_peer_reachable(
    transport: MatrixTransport, target_address: Address, timeout: float = 5.0
) -> None:
    _wait_for_peer_reachability(
        transport=transport,
        target_address=target_address,
        target_reachability=AddressReachability.REACHABLE,
        timeout=timeout,
    )


@pytest.mark.parametrize("matrix_server_count", [2])
@pytest.mark.parametrize("number_of_transports", [2])
def test_matrix_message_sync(matrix_transports: List[MatrixTransport]) -> None:
    transport0, transport1 = matrix_transports
    transport0_messages: Set[Union[Processed, Delivered]] = set()
    transport1_messages: Set[Union[Processed, Delivered]] = set()
    transport0_message_handler = MessageHandler(transport0_messages)
    transport1_message_handler = MessageHandler(transport1_messages)
    raiden_service0 = MockRaidenService(transport0_message_handler)
    raiden_service1 = MockRaidenService(transport1_message_handler)
    raiden_service1.handle_and_track_state_changes = MagicMock()
    transport0.start(raiden_service0, None)
    transport1.start(raiden_service1, None)
    queue_identifier = QueueIdentifier(
        recipient=transport1._raiden_service.address,
        canonical_identifier=factories.UNIT_CANONICAL_ID,
    )
    raiden0_queues: Dict[QueueIdentifier, List[Union[Processed, Delivered]]] = (
        views.get_all_messagequeues(views.state_from_raiden(raiden_service0))
    )
    raiden0_queues[queue_identifier] = []
    for i in range(5):
        message = Processed(message_identifier=MessageID(i), signature=EMPTY_SIGNATURE)
        raiden0_queues[queue_identifier].append(message)
        transport0._raiden_service.sign(message)
        transport0.send_async(
            [MessagesQueue(queue_identifier, [(message, transport1.address_metadata)])]
        )
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
        transport0.send_async(
            [MessagesQueue(queue_identifier, [(message, transport1.address_metadata)])]
        )
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


def test_matrix_message_retry(
    local_matrix_servers: List[str],
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
) -> None:
    """Test the retry mechanism implemented into the matrix client.
    The test creates a transport and sends a message. The receiver
    doesn't respond in time. The retrier should then
    wait for the `retry_interval` duration to pass and send the message
    again.
    """
    partner_address: Address = factories.make_address()
    transport = MatrixTransport(
        config=MatrixTransportConfig(
            retries_before_backoff=retries_before_backoff,
            retry_interval_initial=retry_interval_initial,
            retry_interval_max=retry_interval_max,
            server=local_matrix_servers[0],
            available_servers=[local_matrix_servers[0]],
        ),
        environment=Environment.DEVELOPMENT,
    )
    transport._send_raw = MagicMock()
    raiden_service = MockRaidenService(None)
    transport.start(raiden_service, None)
    transport.log = MagicMock()
    queueid = QueueIdentifier(
        recipient=partner_address, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE
    )
    chain_state = raiden_service.wal.get_current_state()
    retry_queue: Optional[_RetryQueue] = transport._get_retrier(partner_address)
    assert retry_queue, "retry_queue not running"
    message = Processed(message_identifier=MessageID(0), signature=EMPTY_SIGNATURE)
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


@pytest.mark.parametrize("matrix_server_count", [3])
@pytest.mark.parametrize("number_of_transports", [2])
def test_matrix_transport_handles_metadata(
    matrix_transports: List[MatrixTransport],
) -> None:
    transport0, transport1 = matrix_transports
    transport0_messages: Set[Union[Processed, Delivered]] = set()
    transport1_messages: Set[Union[Processed, Delivered]] = set()
    transport0_message_handler = MessageHandler(transport0_messages)
    transport1_message_handler = MessageHandler(transport1_messages)
    raiden_service0 = MockRaidenService(transport0_message_handler)
    raiden_service1 = MockRaidenService(transport1_message_handler)
    raiden_service1.handle_and_track_state_changes = MagicMock()
    transport0.start(raiden_service0, None)
    transport1.start(raiden_service1, None)
    queue_identifier = QueueIdentifier(
        recipient=transport1._raiden_service.address,
        canonical_identifier=factories.UNIT_CANONICAL_ID,
    )
    raiden0_queues: Dict[QueueIdentifier, List[Union[Processed, Delivered]]] = (
        views.get_all_messagequeues(views.state_from_raiden(raiden_service0))
    )
    raiden0_queues[queue_identifier] = []
    correct_metadata: Dict[str, str] = {"user_id": transport1.user_id}
    invalid_metadata: Dict[str, str] = {"user_id": "invalid"}
    no_metadata: Optional[Dict[str, str]] = None
    all_metadata: Tuple[Optional[Dict[str, str]], ...] = (
        correct_metadata,
        invalid_metadata,
        no_metadata,
    )
    num_sends: int = 2
    message_id: int = 0
    for metadata in all_metadata:
        for _ in range(num_sends):
            message: Union[Processed, Delivered] = Processed(
                message_identifier=MessageID(message_id),
                signature=EMPTY_SIGNATURE,
            )
            raiden0_queues[queue_identifier].append(message)
            transport0._raiden_service.sign(message)
            message_queues = [
                MessagesQueue(queue_identifier, [(message, metadata)])
            ]
            transport0.send_async(message_queues)
            message_id += 1
    num_expected_messages: int = num_sends
    with Timeout(TIMEOUT_MESSAGE_RECEIVE):
        while len(transport0_messages) < num_expected_messages:
            gevent.sleep(0.1)
        while len(transport1_messages) < num_expected_messages:
            gevent.sleep(0.1)
    ids = [cast(int, m.message_identifier) for m in transport1_messages]
    delivered_ids = [cast(int, m.delivered_message_identifier) for m in transport0_messages]
    assert sorted(ids) == sorted(delivered_ids)
    assert sorted(ids) == list(range(0, num_sends))
    assert len(transport0_messages) == num_expected_messages
    assert len(transport1_messages) == num_expected_messages
    transport0.stop()
    transport1.stop()


@pytest.mark.parametrize("matrix_server_count", [2])
@pytest.mark.parametrize("number_of_transports", [3])
def test_matrix_cross_server_with_load_balance(
    matrix_transports: List[MatrixTransport],
) -> None:
    transport0, transport1, transport2 = matrix_transports
    received_messages0: Set[Union[Processed, Delivered]] = set()
    received_messages1: Set[Union[Processed, Delivered]] = set()
    received_messages2: Set[Union[Processed, Delivered]] = set()
    message_handler0 = MessageHandler(received_messages0)
    message_handler1 = MessageHandler(received_messages1)
    message_handler2 = MessageHandler(received_messages2)
    raiden_service0 = MockRaidenService(message_handler0)
    raiden_service1 = MockRaidenService(message_handler1)
    raiden_service2 = MockRaidenService(message_handler2)
    transport0.start(raiden_service0, "")
    transport1.start(raiden_service1, "")
    transport2.start(raiden_service2, "")
    assert ping_pong_message_success(transport0, transport1)
    assert ping_pong_message_success(transport0, transport2)
    assert ping_pong_message_success(transport1, transport0)
    assert ping_pong_message_success(transport1, transport2)
    assert ping_pong_message_success(transport2, transport0)
    assert ping_pong_message_success(transport2, transport1)


@pytest.mark.parametrize("device_id", [DeviceIDs.PFS, DeviceIDs.MS])
def test_matrix_broadcast(
    matrix_transports: List[MatrixTransport],
    services: List[str],
    device_id: DeviceIDs,
) -> None:
    transport: MatrixTransport = matrix_transports[0]
    matrix_api = transport._client.api
    matrix_api.send_to_device = MagicMock(autospec=True)
    transport.start(MockRaidenService(None), "")
    gevent.idle()
    sent_messages: List[Union[Processed, Delivered]] = []
    for i in range(5):
        message: Union[Processed, Delivered] = Processed(
            message_identifier=MessageID(i), signature=EMPTY_SIGNATURE
        )
        transport._raiden_service.sign(message)
        sent_messages.append(message)
        transport.broadcast(message, device_id=device_id)
    transport._schedule_new_greenlet(transport._broadcast_worker)
    gevent.idle()
    messages = get_to_device_broadcast_messages(
        matrix_api.send_to_device, services, device_id.value
    )
    assert messages == sent_messages


@pytest.mark.parametrize("environment_type", [Environment.DEVELOPMENT])
def test_monitoring_broadcast_messages(
    matrix_transports: List[MatrixTransport],
    environment_type: Environment,
    services: List[str],
    monkeypatch: Callable[..., None],
) -> None:
    """
    Test that RaidenService broadcast RequestMonitoring messages to
    MONITORING_BROADCASTING_ROOM room on newly received balance proofs.
    """
    transport: MatrixTransport = matrix_transports[0]
    matrix_api = transport._client.api
    matrix_api.retry_timeout = 0
    matrix_api.send_to_device = MagicMock(autospec=True)
    raiden_service = MockRaidenService(None)
    raiden_service.config = RaidenConfig(
        chain_id=ChainID(1234),
        environment_type=environment_type,
        services=ServiceConfig(monitoring_enabled=True),
        pfs_config=make_pfs_config(),
    )
    transport.start(raiden_service, None)
    raiden_service.transport = transport
    transport.log = MagicMock()
    balance_proof = factories.create(HOP1_BALANCE_PROOF)
    channel_state = factories.create(
        factories.NettingChannelStateProperties()
    )
    channel_state.our_state.balance_proof = balance_proof
    channel_state.partner_state.balance_proof = balance_proof
    monkeypatch.setattr(
        raiden.transfer.views, "get_channelstate_by_canonical_identifier", lambda *a, **kw: channel_state
    )
    monkeypatch.setattr(
        raiden.transfer.channel, "get_balance", lambda *a, **kw: 123
    )
    raiden_service.default_user_deposit.effective_balance.return_value = MONITORING_REWARD
    update_monitoring_service_from_balance_proof(
        raiden=raiden_service,
        chain_state=None,
        new_balance_proof=balance_proof,
        non_closing_participant=HOP1,
    )
    gevent.idle()
    with gevent.Timeout(2):
        while matrix_api.send_to_device.call_count < 1:
            gevent.idle()
    assert matrix_api.send_to_device.call_count == 1
    messages = get_to_device_broadcast_messages(
        matrix_api.send_to_device, services, DeviceIDs.MS.value
    )
    assert len(messages) == 1
    assert isinstance(messages[0], RequestMonitoring)


@pytest.mark.parametrize("environment_type", [Environment.PRODUCTION])
@pytest.mark.parametrize(
    "channel_balance_dai, expected_messages",
    [
        (MIN_MONITORING_AMOUNT_DAI - 1, 0),
        (MIN_MONITORING_AMOUNT_DAI, 1),
    ],
)
def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(
    matrix_transports: List[MatrixTransport],
    services: List[str],
    monkeypatch: Callable[..., None],
    channel_balance_dai: int,
    expected_messages: int,
    environment_type: Environment,
) -> None:
    """
    Test that in PRODUCTION on DAI and WETH RaidenService broadcast RequestMonitoring messages
    to MONITORING_BROADCASTING_ROOM room on newly received balance proofs only when
    min threshold of channel balance is met
    """
    transport: MatrixTransport = matrix_transports[0]
    matrix_api = transport._client.api
    matrix_api.retry_timeout = 0
    matrix_api.send_to_device = MagicMock(autospec=True)
    raiden_service = MockRaidenService(None)
    raiden_service.config = RaidenConfig(
        chain_id=1234,
        environment_type=environment_type,
        services=ServiceConfig(monitoring_enabled=True),
        pfs_config=make_pfs_config(),
    )
    transport.start(raiden_service, None)
    raiden_service.transport = transport
    transport.log = MagicMock()
    fake_dai_token_network: Address = factories.make_token_network_address()
    HOP1_BALANCE_PROOF_DAI = factories.BalanceProofSignedStateProperties(
        pkey=factories.HOP1_KEY,
        canonical_identifier=factories.create(
            CanonicalIdentifierProperties(token_network_address=fake_dai_token_network)
        ),
    )
    balance_proof = factories.create(HOP1_BALANCE_PROOF_DAI)
    channel_state = factories.create(
        factories.NettingChannelStateProperties(
            canonical_identifier=CanonicalIdentifierProperties(
                token_network_address=fake_dai_token_network
            ),
            our_state=NettingChannelEndStateProperties(balance=channel_balance_dai),
        )
    )
    channel_state.our_state.balance_proof = balance_proof
    channel_state.partner_state.balance_proof = balance_proof
    monkeypatch.setattr(
        raiden.transfer.views,
        "get_channelstate_by_canonical_identifier",
        lambda *a, **kw: channel_state,
    )
    monkeypatch.setattr(
        raiden.transfer.views,
        "get_token_network_address_by_token_address",
        lambda *a, **kw: fake_dai_token_network,
    )
    raiden_service.default_user_deposit.effective_balance.return_value = MONITORING_REWARD
    update_monitoring_service_from_balance_proof(
        raiden=raiden_service,
        chain_state=None,
        new_balance_proof=balance_proof,
        non_closing_participant=HOP1,
    )
    gevent.sleep(2)
    messages = get_to_device_broadcast_messages(
        matrix_api.send_to_device, services, DeviceIDs.MS.value
    )
    assert len(messages) == expected_messages
    if expected_messages >= 1:
        assert isinstance(messages[0], RequestMonitoring)


@pytest.mark.parametrize("matrix_server_count", [1])
def test_pfs_broadcast_messages(
    matrix_transports: List[MatrixTransport],
    services: List[str],
    monkeypatch: Callable[..., None],
) -> None:
    """
    Test that RaidenService broadcasts PFSCapacityUpdate messages to
    all service addresses via to-device multicast
    """
    transport: MatrixTransport = matrix_transports[0]
    matrix_api = transport._client.api
    matrix_api.retry_timeout = 0
    matrix_api.send_to_device = MagicMock(autospec=True)
    raiden_service = MockRaidenService(None)
    raiden_service.config.services.monitoring_enabled = True
    raiden_service.routing_mode = RoutingMode.PFS
    transport.start(raiden_service, None)
    raiden_service.transport = transport
    transport.log = MagicMock()
    balance_proof = factories.create(HOP1_BALANCE_PROOF)
    channel_state = factories.create(
        factories.NettingChannelStateProperties()
    )
    channel_state.our_state.balance_proof = balance_proof
    channel_state.partner_state.balance_proof = balance_proof
    monkeypatch.setattr(
        raiden.transfer.views, "get_channelstate_by_canonical_identifier", lambda *a, **kw: channel_state
    )
    send_pfs_update(
        raiden=raiden_service,
        canonical_identifier=balance_proof.canonical_identifier,
    )
    gevent.idle()
    with gevent.Timeout(2):
        while matrix_api.send_to_device.call_count < 1:
            gevent.idle()
    assert matrix_api.send_to_device.call_count == 1
    channel_state = factories.create(
        factories.NettingChannelStateProperties()
    )
    fee_update: PFSFeeUpdate = PFSFeeUpdate.from_channel_state(channel_state)
    fee_update.sign(raiden_service.signer)
    raiden_service.transport.broadcast(fee_update, device_id=DeviceIDs.PFS)
    with gevent.Timeout(2):
        while matrix_api.send_to_device.call_count < 2:
            gevent.idle()
    assert matrix_api.send_to_device.call_count == 2
    messages = get_to_device_broadcast_messages(
        matrix_api.send_to_device, services, DeviceIDs.PFS.value
    )
    assert len(messages) == 2
    assert isinstance(messages[0], PFSCapacityUpdate)
    assert isinstance(messages[1], PFSFeeUpdate)


@pytest.mark.parametrize("matrix_server_count", [3])
@pytest.mark.parametrize("number_of_transports", [3])
@pytest.mark.parametrize(
    "roaming_peer",
    [pytest.param("high", id="roaming_high"), pytest.param("low", id="roaming_low")],
)
def test_matrix_user_roaming(
    matrix_transports: List[MatrixTransport], roaming_peer: str
) -> None:
    transport0, transport1, transport2 = matrix_transports
    received_messages0: Set[Union[Processed, Delivered]] = set()
    received_messages1: Set[Union[Processed, Delivered]] = set()
    message_handler0 = MessageHandler(received_messages0)
    message_handler1 = MessageHandler(received_messages1)
    reverse_privkey_order: bool = roaming_peer == "low"
    privkey0, privkey1 = make_privkeys_ordered(
        count=2, reverse=reverse_privkey_order
    )
    raiden_service0 = MockRaidenService(
        message_handler0, private_key=privkey0
    )
    raiden_service1 = MockRaidenService(
        message_handler1, private_key=privkey1
    )
    transport0.start(raiden_service0, "")
    transport1.start(raiden_service1, "")
    assert ping_pong_message_success(transport0, transport1)
    transport0.stop()
    transport2.start(raiden_service0, "")
    assert ping_pong_message_success(transport2, transport1)
    transport2.stop()
    transport0.start(raiden_service0, "")
    assert ping_pong_message_success(transport0, transport1)


@pytest.mark.parametrize("matrix_server_count", [3])
@pytest.mark.parametrize("number_of_transports", [6])
@pytest.mark.parametrize(
    "roaming_peer",
    [pytest.param("high", id="roaming_high"), pytest.param("low", id="roaming_low")],
)
@pytest.mark.parametrize("capabilities", [CapabilitiesConfig(to_device=True)])
def test_matrix_multi_user_roaming(
    matrix_transports: List[MatrixTransport],
    roaming_peer: str,
) -> None:
    transport_rs0_0, transport_rs0_1, transport_rs0_2, transport_rs1_0, transport_rs1_1, transport_rs1_2 = matrix_transports
    received_messages0: Set[Union[Processed, Delivered]] = set()
    received_messages1: Set[Union[Processed, Delivered]] = set()
    message_handler0 = MessageHandler(received_messages0)
    message_handler1 = MessageHandler(received_messages1)
    reverse_privkey_order: bool = roaming_peer == "low"
    privkey0, privkey1 = make_privkeys_ordered(
        count=2, reverse=reverse_privkey_order
    )
    raiden_service0 = MockRaidenService(
        message_handler0, private_key=privkey0
    )
    raiden_service1 = MockRaidenService(
        message_handler1, private_key=privkey1
    )
    transport_rs0_0.start(raiden_service0, "")
    transport_rs1_0.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_0, transport_rs1_0)
    transport_rs1_0.stop()
    transport_rs1_1.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_0, transport_rs1_1)
    transport_rs1_1.stop()
    transport_rs1_2.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_0, transport_rs1_2)
    transport_rs0_0.stop()
    transport_rs1_2.stop()
    transport_rs0_1.start(raiden_service0, "")
    transport_rs1_0.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_1, transport_rs1_0)
    transport_rs1_0.stop()
    transport_rs1_1.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_1, transport_rs1_1)
    transport_rs1_1.stop()
    transport_rs1_2.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_1, transport_rs1_2)
    transport_rs0_1.stop()
    transport_rs1_2.stop()
    transport_rs0_2.start(raiden_service0, "")
    transport_rs1_0.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_2, transport_rs1_0)
    transport_rs1_0.stop()
    transport_rs1_1.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_2, transport_rs1_1)
    transport_rs1_1.stop()
    transport_rs1_2.start(raiden_service1, "")
    assert ping_pong_message_success(transport_rs0_2, transport_rs1_2)


def test_populate_services_addresses(
    service_registry_address: Address,
    private_keys: List[str],
    web3: Any,
    contract_manager: Any,
) -> None:
    """
    Test 'populate_services_addresses' parsing addresses from service_registry_contract.
    """
    c1_service_proxy, _ = deploy_service_registry_and_set_urls(
        private_keys=private_keys,
        web3=web3,
        contract_manager=contract_manager,
        service_registry_address=service_registry_address,
    )
    addresses: List[str] = [privatekey_to_address(key) for key in private_keys]
    transport: Mock = Mock()
    populate_services_addresses(
        transport=transport,
        service_registry=c1_service_proxy,
        block_identifier=BLOCK_ID_LATEST,
    )
    registered_services: List[str] = list(transport.update_services_addresses.call_args[0][0].keys())
    assert len(registered_services) == 3
    assert sorted(addresses) == sorted(registered_services)
