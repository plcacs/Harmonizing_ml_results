import gevent
import pytest
from eth_utils import Address
from raiden.settings import MatrixTransportConfig, RaidenConfig, ServiceConfig
from raiden.network.transport.matrix.transport import MatrixTransport
from raiden.utils.typing import ChainID, MessageID
from raiden_contracts.utils.type_aliases import ChainID
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

class MessageHandler:
    def __init__(self, bag: Set[Any]) -> None:
        ...
    
    def on_messages(self, sender: Any, messages: Iterable[Any]) -> None:
        ...

def get_to_device_broadcast_messages(
    to_device_mock: Any,
    expected_receiver_addresses: List[str],
    device_id: str
) -> List[Any]:
    ...

def ping_pong_message_success(transport0: MatrixTransport, transport1: MatrixTransport) -> bool:
    ...

def is_reachable(transport: MatrixTransport, address: Address) -> bool:
    ...

def test_matrix_message_sync(matrix_transports: List[MatrixTransport]) -> None:
    ...

def test_matrix_message_retry(
    local_matrix_servers: Any,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int
) -> None:
    ...

def test_matrix_transport_handles_metadata(matrix_transports: List[MatrixTransport]) -> None:
    ...

def test_matrix_cross_server_with_load_balance(matrix_transports: List[MatrixTransport]) -> None:
    ...

def test_matrix_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None:
    ...

def test_matrix_multi_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None:
    ...

def test_monitoring_broadcast_messages(
    matrix_transports: List[MatrixTransport],
    environment_type: str,
    services: List[str],
    monkeypatch: Any
) -> None:
    ...

def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(
    matrix_transports: List[MatrixTransport],
    services: List[str],
    monkeypatch: Any,
    channel_balance_dai: int,
    expected_messages: int,
    environment_type: str
) -> None:
    ...

def test_pfs_broadcast_messages(
    matrix_transports: List[MatrixTransport],
    services: List[str],
    monkeypatch: Any
) -> None:
    ...

def test_populate_services_addresses(
    service_registry_address: Address,
    private_keys: List[str],
    web3: Any,
    contract_manager: Any
) -> None:
    ...

@pytest.fixture
def num_services() -> int:
    ...

@pytest.fixture
def services(num_services: int, matrix_transports: List[MatrixTransport]) -> List[str]:
    ...

@pytest.fixture
def number_of_transports() -> int:
    ...