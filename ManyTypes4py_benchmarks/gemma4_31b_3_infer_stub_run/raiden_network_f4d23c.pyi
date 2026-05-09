import os
import subprocess
from pathlib import Path
from typing import Generator, List, Optional, Any, Union, Iterable
from raiden.constants import Environment, RoutingMode
from raiden.network.pathfinding import PFSInfo
from raiden.raiden_service import RaidenService
from raiden.settings import CapabilitiesConfig
from raiden.tests.utils.mocks import PFSMock
from raiden.utils.typing import (
    BlockNumber,
    BlockTimeout,
    ChainID,
    MonitoringServiceAddress,
    OneToNAddress,
    Port,
    ServiceRegistryAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkRegistryAddress,
    UserDepositAddress,
)

def timeout(blockchain_type: str) -> int: ...

@pytest.fixture
def routing_mode() -> RoutingMode: ...

@pytest.fixture
def raiden_chain(
    token_addresses: List[TokenAddress],
    token_network_registry_address: TokenNetworkRegistryAddress,
    one_to_n_address: OneToNAddress,
    monitoring_service_address: MonitoringServiceAddress,
    channels_per_node: int,
    deposit: TokenAmount,
    settle_timeout: BlockTimeout,
    chain_id: ChainID,
    blockchain_services: Any,
    reveal_timeout: BlockTimeout,
    retry_interval_initial: int,
    retry_interval_max: int,
    retries_before_backoff: int,
    environment_type: Environment,
    unrecoverable_error_should_crash: bool,
    local_matrix_servers: List[str],
    blockchain_type: str,
    contracts_path: str,
    user_deposit_address: UserDepositAddress,
    logs_storage: str,
    register_tokens: bool,
    start_raiden_apps: bool,
    routing_mode: RoutingMode,
    blockchain_query_interval: int,
    resolver_ports: List[Optional[Port]],
    enable_rest_api: bool,
    port_generator: Any,
    capabilities: CapabilitiesConfig,
) -> Generator[List[RaidenService], None, None]: ...

@pytest.fixture
def resolvers(resolver_ports: List[Optional[Port]]) -> Generator[List[Optional[subprocess.Popen]], None, None]: ...

@pytest.fixture
def adhoc_capability() -> bool: ...

@pytest.fixture
def capabilities(adhoc_capability: bool) -> CapabilitiesConfig: ...

@pytest.fixture
def pfs_mock(
    monkeypatch: Any,
    local_matrix_servers: List[str],
    chain_id: ChainID,
    token_network_registry_address: Optional[TokenNetworkRegistryAddress],
    user_deposit_address: Optional[UserDepositAddress],
) -> PFSMock: ...

@pytest.fixture
def raiden_network(
    token_addresses: List[TokenAddress],
    token_network_registry_address: TokenNetworkRegistryAddress,
    one_to_n_address: OneToNAddress,
    monitoring_service_address: MonitoringServiceAddress,
    channels_per_node: int,
    deposit: TokenAmount,
    settle_timeout: BlockTimeout,
    chain_id: ChainID,
    blockchain_services: Any,
    reveal_timeout: BlockTimeout,
    retry_interval_initial: int,
    retry_interval_max: int,
    retries_before_backoff: int,
    environment_type: Environment,
    unrecoverable_error_should_crash: bool,
    local_matrix_servers: List[str],
    blockchain_type: str,
    contracts_path: str,
    user_deposit_address: UserDepositAddress,
    logs_storage: str,
    register_tokens: bool,
    start_raiden_apps: bool,
    routing_mode: RoutingMode,
    blockchain_query_interval: int,
    resolver_ports: List[Optional[Port]],
    enable_rest_api: bool,
    port_generator: Any,
    capabilities: CapabilitiesConfig,
) -> Generator[List[RaidenService], None, None]: ...

class RestartNode:
    async_result: Optional[Any]
    def __init__(self) -> None: ...
    def link_exception_to(self, result: Any) -> None: ...
    def __call__(self, service: RaidenService) -> None: ...

@pytest.fixture
def restart_node() -> RestartNode: ...