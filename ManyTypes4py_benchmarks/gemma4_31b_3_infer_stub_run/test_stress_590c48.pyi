import gevent
from typing import Any, Generator, Iterator, List, Tuple, Sequence, Optional
from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.utils.typing import Address, TokenAddress, TokenAmount, TokenNetworkAddress, Port

def iwait_and_get(items: Sequence[gevent.Greenlet[Any]]) -> None: ...

def _url_for(apiserver: APIServer, endpoint: str, **kwargs: Any) -> str: ...

def start_apiserver(raiden_app: RaidenService, rest_api_port_number: Port) -> APIServer: ...

def start_apiserver_for_network(raiden_network: Sequence[RaidenService], port_generator: Iterator[Port]) -> List[APIServer]: ...

def restart_app(app: RaidenService, restart_node: Any) -> RaidenService: ...

def restart_network(raiden_network: Sequence[RaidenService], restart_node: Any) -> List[RaidenService]: ...

def restart_network_and_apiservers(
    raiden_network: Sequence[RaidenService], 
    restart_node: Any, 
    api_servers: Sequence[APIServer], 
    port_generator: Iterator[Port]
) -> Tuple[List[RaidenService], List[APIServer]]: ...

def address_from_apiserver(apiserver: APIServer) -> Address: ...

def transfer_and_assert(
    server_from: APIServer, 
    server_to: APIServer, 
    token_address: TokenAddress, 
    identifier: Any, 
    amount: TokenAmount
) -> None: ...

def sequential_transfers(
    server_from: APIServer, 
    server_to: APIServer, 
    number_of_transfers: int, 
    token_address: TokenAddress, 
    identifier_generator: Iterator[Any]
) -> None: ...

def stress_send_serial_transfers(
    rest_apis: Sequence[APIServer], 
    token_address: TokenAddress, 
    identifier_generator: Iterator[Any], 
    deposit: int
) -> None: ...

def stress_send_parallel_transfers(
    rest_apis: Sequence[APIServer], 
    token_address: TokenAddress, 
    identifier_generator: Iterator[Any], 
    deposit: int
) -> None: ...

def stress_send_and_receive_parallel_transfers(
    rest_apis: Sequence[APIServer], 
    token_address: TokenAddress, 
    identifier_generator: Iterator[Any], 
    deposit: int
) -> None: ...

def assert_channels(raiden_network: Sequence[RaidenService], token_network_address: TokenNetworkAddress, deposit: int) -> None: ...

def test_stress(
    raiden_network: Sequence[RaidenService], 
    restart_node: Any, 
    deposit: int, 
    token_addresses: Sequence[TokenAddress], 
    port_generator: Iterator[Port]
) -> None: ...