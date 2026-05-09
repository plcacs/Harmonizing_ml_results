import random
from unittest.mock import MagicMock, Mock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID
import gevent
import pytest
from eth_utils import Address
from gevent import Timeout
from raiden.constants import (
    BLOCK_ID_LATEST,
    EMPTY_SIGNATURE,
    DeviceIDs,
    Environment,
    RoutingMode,
)
from raiden.messages.monitoring_service import RequestMonitoring
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.messages.synchronization import Delivered, Processed
from raiden.network.transport.matrix.transport import (
    MatrixTransport,
    MessagesQueue,
    _RetryQueue,
    populate_services_addresses,
)
from raiden.settings import (
    CapabilitiesConfig,
    MatrixTransportConfig,
    RaidenConfig,
    ServiceConfig,
)
from raiden.storage.serialization.serializer import MessageSerializer
from raiden.tests.utils.factories import (
    CanonicalIdentifierProperties,
    NettingChannelEndStateProperties,
)
from raiden.utils.typing import ChainID, MessageID
from raiden_contracts.utils.type_aliases import ChainID

num_services: Callable[[], int] = ...
services: Callable[[int, List[MatrixTransport]], List[Address]] = ...
number_of_transports: Callable[[], int] = ...

class MessageHandler:
    def __init__(self, bag: Any) -> None:
        ...
    def on_messages(self, _, messages: Any) -> None:
        ...

def get_to_device_broadcast_messages(
    to_device_mock: MagicMock,
    expected_receiver_addresses: List[str],
    device_id: str,
) -> List[Any]:
    ...

def ping_pong_message_success(transport0: MatrixTransport, transport1: MatrixTransport) -> bool:
    ...

def is_reachable(transport: MatrixTransport, address: Address) -> bool:
    ...

def _wait_for_peer_reachability(
    transport: MatrixTransport,
    target_address: Address,
    target_reachability: str,
    timeout: float = ...,
) -> None:
    ...

def wait_for_peer_unreachable(transport: MatrixTransport, target_address: Address, timeout: float = ...) -> None:
    ...

def wait_for_peer_reachable(transport: MatrixTransport, target_address: Address, timeout: float = ...) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_sync(matrix_transports: List[MatrixTransport]) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [3])
def test_matrix_cross_server_with_load_balance(matrix_transports: List[MatrixTransport]) -> None:
    ...

@pytest.mark.parametrize('device_id', (DeviceIDs.PFS, DeviceIDs.MS))
def test_matrix_broadcast(matrix_transports: List[MatrixTransport], services: List[Address], device_id: DeviceIDs) -> None:
    ...

@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
def test_monitoring_broadcast_messages(
    matrix_transports: List[MatrixTransport],
    services: List[Address],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ...

@pytest.mark.parametrize('environment_type', [Environment.PRODUCTION])
@pytest.mark.parametrize('channel_balance_dai, expected_messages', [[MIN_MONITORING_AMOUNT_DAI - 1, 0], [MIN_MONITORING_AMOUNT_DAI, 1]])
def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(
    matrix_transports: List[MatrixTransport],
    services: List[Address],
    monkeypatch: pytest.MonkeyPatch,
    channel_balance_dai: int,
    expected_messages: int,
    environment_type: Environment,
) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [1])
def test_pfs_broadcast_messages(matrix_transports: List[MatrixTransport], services: List[Address], monkeypatch: pytest.MonkeyPatch) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [3])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
def test_matrix_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [6])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
@pytest.mark.parametrize('capabilities', [CapabilitiesConfig(to_device=True)])
def test_matrix_multi_user_roaming(matrix_transports: List[MatrixTransport], roaming_peer: str) -> None:
    ...

def test_populate_services_addresses(
    service_registry_address: Address,
    private_keys: List[str],
    web3: Any,
    contract_manager: Any,
) -> None:
    ...