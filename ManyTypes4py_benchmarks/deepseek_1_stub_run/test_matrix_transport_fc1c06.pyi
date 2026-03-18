```python
import random
from typing import Any, Dict, List, Set, Tuple, Optional, Union, Callable
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

HOP1_BALANCE_PROOF: Any = ...
TIMEOUT_MESSAGE_RECEIVE: int = ...

@pytest.fixture
def num_services() -> Any: ...

@pytest.fixture
def services(num_services: Any, matrix_transports: Any) -> Any: ...

@pytest.fixture
def number_of_transports() -> Any: ...

class MessageHandler:
    def __init__(self, bag: Any) -> None: ...
    def on_messages(self, _: Any, messages: Any) -> None: ...

def get_to_device_broadcast_messages(to_device_mock: Any, expected_receiver_addresses: Any, device_id: Any) -> Any: ...

def ping_pong_message_success(transport0: Any, transport1: Any) -> Any: ...

def is_reachable(transport: Any, address: Any) -> Any: ...

def _wait_for_peer_reachability(transport: Any, target_address: Any, target_reachability: Any, timeout: int = ...) -> Any: ...

def wait_for_peer_unreachable(transport: Any, target_address: Any, timeout: int = ...) -> Any: ...

def wait_for_peer_reachable(transport: Any, target_address: Any, timeout: int = ...) -> Any: ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_sync(matrix_transports: Any) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_message_retry(local_matrix_servers: Any, retry_interval_initial: Any, retry_interval_max: Any, retries_before_backoff: Any) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [2])
def test_matrix_transport_handles_metadata(matrix_transports: Any) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [2])
@pytest.mark.parametrize('number_of_transports', [3])
def test_matrix_cross_server_with_load_balance(matrix_transports: Any) -> None: ...

@pytest.mark.parametrize('device_id', (DeviceIDs.PFS, DeviceIDs.MS))
def test_matrix_broadcast(matrix_transports: Any, services: Any, device_id: Any) -> None: ...

@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
def test_monitoring_broadcast_messages(matrix_transports: Any, environment_type: Any, services: Any, monkeypatch: Any) -> None: ...

@pytest.mark.parametrize('environment_type', [Environment.PRODUCTION])
@pytest.mark.parametrize('channel_balance_dai, expected_messages', [[MIN_MONITORING_AMOUNT_DAI - 1, 0], [MIN_MONITORING_AMOUNT_DAI, 1]])
def test_monitoring_broadcast_messages_in_production_if_bigger_than_threshold(matrix_transports: Any, services: Any, monkeypatch: Any, channel_balance_dai: Any, expected_messages: Any, environment_type: Any) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [1])
def test_pfs_broadcast_messages(matrix_transports: Any, services: Any, monkeypatch: Any) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [3])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
def test_matrix_user_roaming(matrix_transports: Any, roaming_peer: Any) -> None: ...

@pytest.mark.parametrize('matrix_server_count', [3])
@pytest.mark.parametrize('number_of_transports', [6])
@pytest.mark.parametrize('roaming_peer', [pytest.param('high', id='roaming_high'), pytest.param('low', id='roaming_low')])
@pytest.mark.parametrize('capabilities', [CapabilitiesConfig(to_device=True)])
def test_matrix_multi_user_roaming(matrix_transports: Any, roaming_peer: Any) -> None: ...

def test_populate_services_addresses(service_registry_address: Any, private_keys: Any, web3: Any, contract_manager: Any) -> None: ...
```