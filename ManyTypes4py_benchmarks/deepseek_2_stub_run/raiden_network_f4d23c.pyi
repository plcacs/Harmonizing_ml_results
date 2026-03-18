```python
import os
import subprocess
from pathlib import Path
import gevent
import pytest
from gevent.event import AsyncResult
from raiden.constants import Environment, RoutingMode
from raiden.network.pathfinding import PFSInfo
from raiden.raiden_service import RaidenService
from raiden.settings import CapabilitiesConfig
from raiden.tests.utils import factories
from raiden.tests.utils.mocks import PFSMock
from raiden.tests.utils.network import (
    CHAIN,
    BlockchainServices,
    create_all_channels_for_network,
    create_apps,
    create_network_channels,
    create_sequential_channels,
    parallel_start_apps,
    wait_for_alarm_start,
    wait_for_channels,
    wait_for_token_networks,
)
from raiden.tests.utils.tests import shutdown_apps_and_cleanup_tasks
from raiden.tests.utils.transport import ParsedURL
from raiden.utils.formatting import to_canonical_address
from raiden.utils.typing import (
    BlockNumber,
    BlockTimeout,
    ChainID,
    Iterable,
    Iterator,
    List,
    MonitoringServiceAddress,
    OneToNAddress,
    Optional,
    Port,
    ServiceRegistryAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkRegistryAddress,
    UserDepositAddress,
)
from typing import Any

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
    blockchain_services: BlockchainServices,
    reveal_timeout: BlockTimeout,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
    environment_type: Environment,
    unrecoverable_error_should_crash: bool,
    local_matrix_servers: List[ParsedURL],
    blockchain_type: str,
    contracts_path: Path,
    user_deposit_address: UserDepositAddress,
    logs_storage: str,
    register_tokens: bool,
    start_raiden_apps: bool,
    routing_mode: RoutingMode,
    blockchain_query_interval: float,
    resolver_ports: List[Optional[Port]],
    enable_rest_api: bool,
    port_generator: Iterator[Port],
    capabilities: CapabilitiesConfig,
) -> Iterator[List[RaidenService]]: ...

@pytest.fixture
def resolvers(
    resolver_ports: List[Optional[Port]],
) -> Iterator[List[Optional[subprocess.Popen]]]: ...

@pytest.fixture
def adhoc_capability() -> bool: ...

@pytest.fixture
def capabilities(adhoc_capability: bool) -> CapabilitiesConfig: ...

@pytest.fixture
def pfs_mock(
    monkeypatch: Any,
    local_matrix_servers: List[ParsedURL],
    chain_id: ChainID,
    token_network_registry_address: TokenNetworkRegistryAddress,
    user_deposit_address: UserDepositAddress,
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
    blockchain_services: BlockchainServices,
    reveal_timeout: BlockTimeout,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
    environment_type: Environment,
    unrecoverable_error_should_crash: bool,
    local_matrix_servers: List[ParsedURL],
    blockchain_type: str,
    contracts_path: Path,
    user_deposit_address: UserDepositAddress,
    logs_storage: str,
    register_tokens: bool,
    start_raiden_apps: bool,
    routing_mode: RoutingMode,
    blockchain_query_interval: float,
    resolver_ports: List[Optional[Port]],
    enable_rest_api: bool,
    port_generator: Iterator[Port],
    capabilities: CapabilitiesConfig,
) -> Iterator[List[RaidenService]]: ...

class RestartNode:
    async_result: Optional[Any]
    def __init__(self) -> None: ...
    def link_exception_to(self, result: Any) -> None: ...
    def __call__(self, service: Any) -> None: ...

@pytest.fixture
def restart_node() -> RestartNode: ...
```