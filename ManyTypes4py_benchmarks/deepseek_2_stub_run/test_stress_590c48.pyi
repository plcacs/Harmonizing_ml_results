```python
import time
from http import HTTPStatus
from itertools import count
from typing import Any, Generator, List, Sequence, Tuple
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

def iwait_and_get(items: Any) -> None: ...

def _url_for(apiserver: Any, endpoint: str, **kwargs: Any) -> Any: ...

def start_apiserver(raiden_app: Any, rest_api_port_number: Any) -> APIServer: ...

def start_apiserver_for_network(raiden_network: Sequence[Any], port_generator: Iterator[Any]) -> List[APIServer]: ...

def restart_app(app: Any, restart_node: RestartNode) -> RaidenService: ...

def restart_network(raiden_network: Sequence[Any], restart_node: RestartNode) -> List[RaidenService]: ...

def restart_network_and_apiservers(raiden_network: Sequence[Any], restart_node: RestartNode, api_servers: Sequence[Any], port_generator: Iterator[Any]) -> Tuple[List[RaidenService], List[APIServer]]: ...

def address_from_apiserver(apiserver: Any) -> Any: ...

def transfer_and_assert(server_from: Any, server_to: Any, token_address: TokenAddress, identifier: int, amount: TokenAmount) -> None: ...

def sequential_transfers(server_from: Any, server_to: Any, number_of_transfers: int, token_address: TokenAddress, identifier_generator: Iterator[int]) -> None: ...

def stress_send_serial_transfers(rest_apis: Sequence[Any], token_address: TokenAddress, identifier_generator: Iterator[int], deposit: int) -> None: ...

def stress_send_parallel_transfers(rest_apis: Sequence[Any], token_address: TokenAddress, identifier_generator: Iterator[int], deposit: int) -> None: ...

def stress_send_and_receive_parallel_transfers(rest_apis: Sequence[Any], token_address: TokenAddress, identifier_generator: Iterator[int], deposit: int) -> None: ...

def assert_channels(raiden_network: Sequence[Any], token_network_address: TokenNetworkAddress, deposit: int) -> None: ...

@pytest.mark.skip(reason='flaky, see https://github.com/raiden-network/raiden/issues/4803')
@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('number_of_tokens', [1])
@pytest.mark.parametrize('channels_per_node', [2])
@pytest.mark.parametrize('deposit', [2])
@pytest.mark.parametrize('reveal_timeout', [15])
@pytest.mark.parametrize('settle_timeout', [120])
def test_stress(raiden_network: Any, restart_node: RestartNode, deposit: int, token_addresses: Sequence[TokenAddress], port_generator: Iterator[Any]) -> None: ...
```