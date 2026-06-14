import structlog
from itertools import count
from typing import Sequence

from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.tests.integration.fixtures.raiden_network import RestartNode
from raiden.utils.typing import (
    Address,
    Iterator,
    List,
    Port,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    Tuple,
)

log: structlog.stdlib.BoundLogger

def iwait_and_get(items: Sequence[gevent.Greenlet]) -> None: ...
def _url_for(apiserver: APIServer, endpoint: str, **kwargs: object) -> str: ...
def start_apiserver(raiden_app: RaidenService, rest_api_port_number: Port) -> APIServer: ...
def start_apiserver_for_network(
    raiden_network: List[RaidenService], port_generator: Iterator[Port]
) -> List[APIServer]: ...
def restart_app(app: RaidenService, restart_node: RestartNode) -> RaidenService: ...
def restart_network(
    raiden_network: List[RaidenService], restart_node: RestartNode
) -> List[RaidenService]: ...
def restart_network_and_apiservers(
    raiden_network: List[RaidenService],
    restart_node: RestartNode,
    api_servers: List[APIServer],
    port_generator: Iterator[Port],
) -> Tuple[List[RaidenService], List[APIServer]]: ...
def address_from_apiserver(apiserver: APIServer) -> Address: ...
def transfer_and_assert(
    server_from: APIServer,
    server_to: APIServer,
    token_address: TokenAddress,
    identifier: int,
    amount: TokenAmount,
) -> None: ...
def sequential_transfers(
    server_from: APIServer,
    server_to: APIServer,
    number_of_transfers: int,
    token_address: TokenAddress,
    identifier_generator: Iterator[int],
) -> None: ...
def stress_send_serial_transfers(
    rest_apis: List[APIServer],
    token_address: TokenAddress,
    identifier_generator: Iterator[int],
    deposit: int,
) -> None: ...
def stress_send_parallel_transfers(
    rest_apis: List[APIServer],
    token_address: TokenAddress,
    identifier_generator: Iterator[int],
    deposit: int,
) -> None: ...
def stress_send_and_receive_parallel_transfers(
    rest_apis: List[APIServer],
    token_address: TokenAddress,
    identifier_generator: Iterator[int],
    deposit: int,
) -> None: ...
def assert_channels(
    raiden_network: List[RaidenService],
    token_network_address: TokenNetworkAddress,
    deposit: int,
) -> None: ...
def test_stress(
    raiden_network: List[RaidenService],
    restart_node: RestartNode,
    deposit: int,
    token_addresses: List[TokenAddress],
    port_generator: Iterator[Port],
) -> None: ...