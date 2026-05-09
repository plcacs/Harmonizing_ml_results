import gevent
import pytest
from unittest.mock import MagicMock
from eth_utils import to_normalized_address
from raiden.utils.typing import Address
from raiden_contracts.utils.type_aliases import ChainID
from raiden.network.transport.matrix.transport import MatrixTransport, MessagesQueue, _RetryQueue
from raiden.services import send_pfs_update, update_monitoring_service_from_balance_proof
from raiden.messages.monitoring_service import RequestMonitoring
from raiden.messages.path_finding_service import PFSCapacityUpdate, PFSFeeUpdate
from raiden.messages.synchronization import Delivered, Processed
from raiden.tests.utils.factories import HOP1
from raiden.utils.typing import MessageID

class MessageHandler:
    def __init__(self, bag: set):
        ...

    def on_messages(self, sender: str, messages: list):
        ...

def get_to_device_broadcast_messages(to_device_mock: MagicMock, expected_receiver_addresses: list[str], device_id: str) -> list:
    ...

def ping_pong_message_success(transport0: MatrixTransport, transport1: MatrixTransport) -> bool:
    ...

def is_reachable(transport: MatrixTransport, address: str) -> bool:
    ...

def _wait_for_peer_reachability(transport: MatrixTransport, target_address: str, target_reachability: AddressReachability, timeout: int = 5) -> None:
    ...

def wait_for_peer_unreachable(transport: MatrixTransport, target_address: str, timeout: int = 5) -> None:
    ...

def wait_for_peer_reachable(transport: MatrixTransport, target_address: str, timeout: int = 5) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_sync(matrix_transports: list[MatrixTransport]) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_retry(local_matrix_servers: list[str], retry_interval_initial: int, retry_interval_max: int, retries_before_backoff: int) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_transport_handles_metadata(matrix_transports: list[MatrixTransport]) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [3])
def test_matrix_cross_server_with_load_balance(matrix_transports: list[MatrixTransport]) -> None:
    ...

@pytest.mark.parametrize('device_id', (DeviceIDs.PFS, DeviceIDs.MS))
def test_matrix_broadcast(matrix_transports: list[MatrixTransport], services: list[str], device_id: DeviceIDs) -> None:
    ...

@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
def test_monitoring_broadcast_messages(matrix_transports: list[MatrixTransport], services: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    ...

@pytest.mark.parametrize('environment_type', [Environment.PRODUCTION])
@pytest.mark.parametrize('channel_balance_dai, expected_messages', [[MIN_MONITORING_AMOUNT_DAI - 1, 0], [MIN_MONITORING_AMOUNT_DAI, 1]])
def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(matrix_transports: list[MatrixTransport], services: list[str], monkeypatch: pytest.MonkeyPatch, channel_balance_dai: int, expected_messages: int) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [1])
def test_pfs_broadcast_messages(matrix_transports: list[MatrixTransport], services: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [3])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
def test_matrix_user_roaming(matrix_transports: list[MatrixTransport], roaming_peer: str) -> None:
    ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [6])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
@pytest.mark.parametrize('capabilities', [CapabilitiesConfig(to_device=True)])
def test_matrix_multi_user_roaming(matrix_transports: list[MatrixTransport], roaming_peer: str) -> None:
    ...

def test_populate_services_addresses(service_registry_address: str, private_keys: list[str], web3: str, contract_manager: str) -> None:
    ...