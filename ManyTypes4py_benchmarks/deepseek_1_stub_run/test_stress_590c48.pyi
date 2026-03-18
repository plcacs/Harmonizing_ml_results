```python
import time
from http import HTTPStatus
from itertools import count
from typing import Any, Generator, Iterable, List, Sequence, Tuple
import gevent
import grequests
import pytest
import structlog
from eth_utils import to_canonical_address
from flask import url_for
from raiden.api.python import RaidenAPI
from raiden.api.rest import APIServer, RestAPI
from raiden.constants import RoutingMode
from raiden.message_handler import MessageHandler
from raiden.network.transport import MatrixTransport
from raiden.raiden_event_handler import RaidenEventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import RestApiConfig
from raiden.tests.integration.api.utils import wait_for_listening_port
from raiden.tests.integration.fixtures.raiden_network import RestartNode
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.tests.utils.transfer import assert_synced_channel_state, wait_assert, watch_for_unlock_failures
from raiden.transfer import views
from raiden.ui.startup import RaidenBundle
from raiden.utils.formatting import to_checksum_address
from raiden.utils.typing import Address, BlockNumber, Host, Iterator, Port, TokenAddress, TokenAmount, TokenNetworkAddress

log: Any = ...

def iwait_and_get(items: Iterable[gevent.Greenlet]) -> None: ...

def _url_for(apiserver: APIServer, endpoint: str, **kwargs: Any) -> str: ...

def start_apiserver(raiden_app: RaidenService, rest_api_port_number: Port) -> APIServer: ...

def start_apiserver_for_network(raiden_network: Sequence[RaidenService], port_generator: Iterator[Port]) -> List[APIServer]: ...

def restart_app(app: RaidenService, restart_node: RestartNode) -> RaidenService: ...

def restart_network(raiden_network: Sequence[RaidenService], restart_node: RestartNode) -> List[RaidenService]: ...

def restart_network_and_apiservers(
    raiden_network: Sequence[RaidenService],
    restart_node: RestartNode,
    api_servers: Sequence[APIServer],
    port_generator: Iterator[Port]
) -> Tuple[List[RaidenService], List[APIServer]]: ...

def address_from_apiserver(apiserver: APIServer) -> Address: ...

def transfer_and_assert(
    server_from: APIServer,
    server_to: APIServer,
    token_address: TokenAddress,
    identifier: int,
    amount: TokenAmount
) -> None: ...

def sequential_transfers(
    server_from: APIServer,
    server_to: APIServer,
    number_of_transfers: int,
    token_address: TokenAddress,
    identifier_generator: Iterator[int]
) -> None: ...

def stress_send_serial_transfers(
    rest_apis: Sequence[APIServer],
    token_address: TokenAddress,
    identifier_generator: Iterator[int],
    deposit: int
) -> None: ...

def stress_send_parallel_transfers(
    rest_apis: Sequence[APIServer],
    token_address: TokenAddress,
    identifier_generator: Iterator[int],
    deposit: int
) -> None: ...

def stress_send_and_receive_parallel_transfers(
    rest_apis: Sequence[APIServer],
    token_address: TokenAddress,
    identifier_generator: Iterator[int],
    deposit: int
) -> None: ...

def assert_channels(
    raiden_network: Sequence[RaidenService],
    token_network_address: TokenNetworkAddress,
    deposit: int
) -> None: ...

@pytest.mark.skip(reason='flaky, see https://github.com/raiden-network/raiden/issues/4803')
@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('number_of_tokens', [1])
@pytest.mark.parametrize('channels_per_node', [2])
@pytest.mark.parametrize('deposit', [2])
@pytest.mark.parametrize('reveal_timeout', [15])
@pytest.mark.parametrize('settle_timeout', [120])
def test_stress(
    raiden_network: List[RaidenService],
    restart_node: RestartNode,
    deposit: int,
    token_addresses: List[TokenAddress],
    port_generator: Iterator[Port]
) -> None: ...
```