import random
from typing import Any, List, Optional, Set
from unittest.mock import MagicMock, Mock

import pytest

from raiden.constants import DeviceIDs, Environment
from raiden.messages.synchronization import Delivered, Processed
from raiden.network.transport.matrix.transport import MatrixTransport, MessagesQueue
from raiden.network.transport.matrix.utils import AddressReachability
from raiden.settings import CapabilitiesConfig
from raiden.tests.utils.factories import BalanceProofSignedStateProperties
from raiden.utils.typing import Address, MessageID

HOP1_BALANCE_PROOF: BalanceProofSignedStateProperties
TIMEOUT_MESSAGE_RECEIVE: int

@pytest.fixture
def num_services() -> int: ...

@pytest.fixture
def services(num_services: int, matrix_transports: list[MatrixTransport]) -> list[str]: ...

@pytest.fixture
def number_of_transports() -> int: ...

class MessageHandler:
    bag: Set[Any]
    def __init__(self, bag: Set[Any]) -> None: ...
    def on_messages(self, _: Any, messages: Any) -> None: ...

def get_to_device_broadcast_messages(
    to_device_mock: MagicMock,
    expected_receiver_addresses: list[str],
    device_id: str,
) -> list[Any]: ...

def ping_pong_message_success(
    transport0: MatrixTransport,
    transport1: MatrixTransport,
) -> bool: ...

def is_reachable(transport: MatrixTransport, address: Address) -> None: ...

def _wait_for_peer_reachability(
    transport: MatrixTransport,
    target_address: Address,
    target_reachability: AddressReachability,
    timeout: int = ...,
) -> None: ...

def wait_for_peer_unreachable(
    transport: MatrixTransport,
    target_address: Address,
    timeout: int = ...,
) -> None: ...

def wait_for_peer_reachable(
    transport: MatrixTransport,
    target_address: Address,
    timeout: int = ...,
) -> None: ...

def test_matrix_message_sync(matrix_transports: list[MatrixTransport]) -> None: ...

def test_matrix_message_retry(
    local_matrix_servers: list[str],
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
) -> None: ...

def test_matrix_transport_handles_metadata(matrix_transports: list[MatrixTransport]) -> None: ...

def test_matrix_cross_server_with_load_balance(matrix_transports: list[MatrixTransport]) -> None: ...

def test_matrix_broadcast(
    matrix_transports: list[MatrixTransport],
    services: list[str],
    device_id: DeviceIDs,
) -> None: ...

def test_monitoring_broadcast_messages(
    matrix_transports: list[MatrixTransport],
    environment_type: Environment,
    services: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None: ...

def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(
    matrix_transports: list[MatrixTransport],
    services: list[str],
    monkeypatch: pytest.MonkeyPatch,
    channel_balance_dai: int,
    expected_messages: int,
    environment_type: Environment,
) -> None: ...

def test_pfs_broadcast_messages(
    matrix_transports: list[MatrixTransport],
    services: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None: ...

def test_matrix_user_roaming(
    matrix_transports: list[MatrixTransport],
    roaming_peer: str,
) -> None: ...

def test_matrix_multi_user_roaming(
    matrix_transports: list[MatrixTransport],
    roaming_peer: str,
) -> None: ...

def test_populate_services_addresses(
    service_registry_address: Any,
    private_keys: list[bytes],
    web3: Any,
    contract_manager: Any,
) -> None: ...