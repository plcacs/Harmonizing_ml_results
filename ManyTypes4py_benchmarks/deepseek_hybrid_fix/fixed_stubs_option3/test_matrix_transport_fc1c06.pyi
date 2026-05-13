from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
from web3 import Web3
from web3.contract import Contract

HOP1_BALANCE_PROOF: Any = factories.BalanceProofSignedStateProperties(pkey=factories.HOP1_KEY)

TIMEOUT_MESSAGE_RECEIVE: int = 15

@pytest.fixture
def num_services() -> int: ...

@pytest.fixture
def services(num_services: int, matrix_transports: List[MatrixTransport]) -> List[str]: ...

@pytest.fixture
def number_of_transports() -> int: ...

class MessageHandler:
    def __init__(self, bag: Set[Any]) -> None: ...
    def on_messages(self, _: Any, messages: List[Any]) -> None: ...

def get_to_device_broadcast_messages(
    to_device_mock: MagicMock,
    expected_receiver_addresses: List[str],
    device_id: DeviceIDs,
) -> List[Union[Processed, RequestMonitoring, PFSCapacityUpdate, PFSFeeUpdate]]: ...

def ping_pong_message_success(transport0: MatrixTransport, transport1: MatrixTransport) -> bool: ...

def is_reachable(transport: MatrixTransport, address: Address) -> bool: ...

def _wait_for_peer_reachability(
    transport: MatrixTransport,
    target_address: Address,
    target_reachability: AddressReachability,
    timeout: int = 5,
) -> None: ...

def wait_for_peer_unreachable(transport: MatrixTransport, target_address: Address, timeout: int = 5) -> None: ...

def wait_for_peer_reachable(transport: MatrixTransport, target_address: Address, timeout: int = 5) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_sync(matrix_transports: List[MatrixTransport]) -> None: ...

def test_matrix_message_retry(
    local_matrix_servers: List[str],
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_transport_handles_metadata(matrix_transports: List[MatrixTransport]) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [3])
def test_matrix_cross_server_with_load_balance(matrix_transports: List[MatrixTransport]) -> None: ...

@pytest.mark.parametrize('device_id', (DeviceIDs.PFS, DeviceIDs.MS))
def test_matrix_broadcast(matrix_transports: List[MatrixTransport], services: List[str], device_id: DeviceIDs) -> None: ...

@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
def test_monitoring_broadcast_messages(
    matrix_transports: List[MatrixTransport],
    environment_type: Environment,
    services: List[str],
    monkeypatch: Any,
) -> None: ...

@pytest.mark.parametrize('environment_type', [Environment.PRODUCTION])
@pytest.mark.parametrize(
    'channel_balance_dai, expected_messages',
    [[MIN_MONITORING_AMOUNT_DAI - 1, 0], [MIN_MONITORING_AMOUNT_DAI, 1]],
)
def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(
    matrix_transports: List[MatrixTransport],
    services: List[str],
    monkeypatch: Any,
    channel_balance_dai: int,
    expected_messages: int,
    environment_type: Environment,
) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [1])
def test_pfs_broadcast_messages(
    matrix_transports: List[MatrixTransport],
    services: List[str],
    monkeypatch: Any,
) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [3])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
def test_matrix_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [6])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
@pytest.mark.parametrize('capabilities', [CapabilitiesConfig(to_device=True)])
def test_matrix_multi_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None: ...

def test_populate_services_addresses(
    service_registry_address: Address,
    private_keys: List[bytes],
    web3: Web3,
    contract_manager: Any,
) -> None: ...