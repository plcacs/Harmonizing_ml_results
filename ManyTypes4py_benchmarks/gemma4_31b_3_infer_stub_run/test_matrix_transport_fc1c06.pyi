import random
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Iterable
from unittest.mock import MagicMock, Mock
import pytest
from eth_utils import Address as EthAddress
from raiden.constants import DeviceIDs, Environment, RoutingMode
from raiden.messages.monitoring_service import RequestMonitoring
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.messages.synchronization import Delivered, Processed
from raiden.network.transport.matrix.transport import MatrixTransport, MessagesQueue
from raiden.network.transport.matrix.utils import AddressReachability
from raiden.settings import CapabilitiesConfig, MatrixTransportConfig, RaidenConfig, ServiceConfig
from raiden.transfer.identifiers import QueueIdentifier
from raiden.utils.typing import Address, MessageID
from raiden_contracts.utils.type_aliases import ChainID

HOP1_BALANCE_PROOF: Any = ...
TIMEOUT_MESSAGE_RECEIVE: int = ...

@pytest.fixture
def num_services() -> int: ...

@pytest.fixture
def services(num_services: int, matrix_transports: List[MatrixTransport]) -> List[Address]: ...

@pytest.fixture
def number_of_transports() -> int: ...

class MessageHandler:
    def __init__(self, bag: Set[Any]) -> None: ...
    def on_messages(self, _: Any, messages: Iterable[Any]) -> None: ...

def get_to_device_broadcast_messages(
    to_device_mock: Mock, 
    expected_receiver_addresses: List[Address], 
    device_id: str
) -> List[Any]: ...

def ping_pong_message_success(transport0: MatrixTransport, transport1: MatrixTransport) -> bool: ...

def is_reachable(transport: MatrixTransport, address: Address) -> bool: ...

def _wait_for_peer_reachability(
    transport: MatrixTransport, 
    target_address: Address, 
    target_reachability: AddressReachability, 
    timeout: int = 5
) -> None: ...

def wait_for_peer_unreachable(transport: MatrixTransport, target_address: Address, timeout: int = 5) -> None: ...

def wait_for_peer_reachable(transport: MatrixTransport, target_address: Address, timeout: int = 5) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_sync(matrix_transports: List[MatrixTransport]) -> None: ...

def test_matrix_message_retry(
    local_matrix_servers: List[Any], 
    retry_interval_initial: int, 
    retry_interval_max: int, 
    retries_before_backoff: int
) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_transport_handles_metadata(matrix_transports: List[MatrixTransport]) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [3])
def test_matrix_cross_server_with_load_balance(matrix_transports: List[MatrixTransport]) -> None: ...

@pytest.mark.parametrize('device_id', (DeviceIDs.PFS, DeviceIDs.MS))
def test_matrix_broadcast(matrix_transports: List[MatrixTransport], services: List[Address], device_id: DeviceIDs) -> None: ...

@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
def test_monitoring_broadcast_messages(
    matrix_transports: List[MatrixTransport], 
    environment_type: Environment, 
    services: List[Address], 
    monkeypatch: Any
) -> None: ...

@pytest.mark.parametrize('environment_type', [Environment.PRODUCTION])
@pytest.mark.parametrize('channel_balance_dai, expected_messages', [[0, 0], [0, 0]]) # Simplified for stub
def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(
    matrix_transports: List[MatrixTransport], 
    services: List[Address], 
    monkeypatch: Any, 
    channel_balance_dai: int, 
    expected_messages: int, 
    environment_type: Environment
) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [1])
def test_pfs_broadcast_messages(matrix_transports: List[MatrixTransport], services: List[Address], monkeypatch: Any) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [3])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high'), pytest.param('low')])
def test_matrix_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [6])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high'), pytest.param('low')])
@pytest.mark.parametrize('capabilities', [CapabilitiesConfig(to_device=True)])
def test_matrix_multi_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None: ...

def test_populate_services_addresses(
    service_registry_address: Address, 
    private_keys: List[Any], 
    web3: Any, 
    contract_manager: Any
) -> None: ...