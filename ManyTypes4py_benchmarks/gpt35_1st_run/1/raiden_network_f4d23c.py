from typing import List, Optional, Iterator, Iterable
from raiden.constants import RoutingMode
from raiden.network.pathfinding import PFSInfo
from raiden.raiden_service import RaidenService
from raiden.settings import CapabilitiesConfig
from raiden.tests.utils import factories
from raiden.tests.utils.mocks import PFSMock
from raiden.tests.utils.network import CHAIN, BlockchainServices, create_all_channels_for_network, create_apps, create_network_channels, create_sequential_channels, parallel_start_apps, wait_for_alarm_start, wait_for_channels, wait_for_token_networks
from raiden.tests.utils.tests import shutdown_apps_and_cleanup_tasks
from raiden.tests.utils.transport import ParsedURL
from raiden.utils.formatting import to_canonical_address
from raiden.utils.typing import BlockNumber, TokenAmount, TokenNetworkRegistryAddress, UserDepositAddress

def timeout(blockchain_type: str) -> int:
    return 120 if blockchain_type == 'parity' else 30

def routing_mode() -> RoutingMode:
    return RoutingMode.PFS

def raiden_chain(token_addresses: List[str], token_network_registry_address: str, one_to_n_address: str, monitoring_service_address: str, channels_per_node: int, deposit: int, settle_timeout: int, chain_id: int, blockchain_services: BlockchainServices, reveal_timeout: int, retry_interval_initial: int, retry_interval_max: int, retries_before_backoff: int, environment_type: Environment, unrecoverable_error_should_crash: bool, local_matrix_servers: List[str], blockchain_type: str, contracts_path: str, user_deposit_address: str, logs_storage: str, register_tokens: bool, start_raiden_apps: bool, routing_mode: RoutingMode, blockchain_query_interval: int, resolver_ports: List[Optional[int]], enable_rest_api: bool, port_generator: Iterator[int], capabilities: CapabilitiesConfig) -> Iterator[List[RaidenService]:
    ...

def resolvers(resolver_ports: List[Optional[int]]) -> Iterator[List[Optional[subprocess.Popen]]]:
    ...

def adhoc_capability() -> bool:
    return False

def capabilities(adhoc_capability: bool) -> CapabilitiesConfig:
    ...

def pfs_mock(monkeypatch, local_matrix_servers: List[str], chain_id: int, token_network_registry_address: str, user_deposit_address: str) -> PFSMock:
    ...

def raiden_network(token_addresses: List[str], token_network_registry_address: str, one_to_n_address: str, monitoring_service_address: str, channels_per_node: int, deposit: int, settle_timeout: int, chain_id: int, blockchain_services: BlockchainServices, reveal_timeout: int, retry_interval_initial: int, retry_interval_max: int, retries_before_backoff: int, environment_type: Environment, unrecoverable_error_should_crash: bool, local_matrix_servers: List[str], blockchain_type: str, contracts_path: str, user_deposit_address: str, logs_storage: str, register_tokens: bool, start_raiden_apps: bool, routing_mode: RoutingMode, blockchain_query_interval: int, resolver_ports: List[Optional[int]], enable_rest_api: bool, port_generator: Iterator[int], capabilities: CapabilitiesConfig) -> Iterator[List[RaidenService]]:
    ...

class RestartNode:

    def __init__(self):
        self.async_result: Optional[AsyncResult] = None

    def link_exception_to(self, result: AsyncResult) -> None:
        ...

    def __call__(self, service: RaidenService) -> None:
        ...

def restart_node() -> RestartNode:
    return RestartNode()
