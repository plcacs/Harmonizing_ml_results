```pyi
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
from raiden.tests.utils.network import CHAIN, BlockchainServices, create_all_channels_for_network, create_apps, create_network_channels, create_sequential_channels, parallel_start_apps, wait_for_alarm_start, wait_for_channels, wait_for_token_networks
from raiden.tests.utils.tests import shutdown_apps_and_cleanup_tasks
from raiden.tests.utils.transport import ParsedURL
from raiden.utils.formatting import to_canonical_address
from raiden.utils.typing import BlockNumber, BlockTimeout, ChainID, Iterable, Iterator, List, MonitoringServiceAddress, OneToNAddress, Optional, Port, ServiceRegistryAddress, TokenAddress, TokenAmount, TokenNetworkRegistryAddress, UserDepositAddress
from typing import Any, Generator

def timeout(blockchain_type: str) -> int: ...

@pytest.fixture
def routing_mode() -> RoutingMode: ...

@pytest.fixture
def raiden_chain(
    token_addresses: Any,
    token_network_registry_address: Any,
    one_to_n_address: Any,
    monitoring_service_address: Any,
    channels_per_node: Any,
    deposit: Any,
    settle_timeout: Any,
    chain_id: Any,
    blockchain_services: Any,
    reveal_timeout: Any,
    retry_interval_initial: Any,
    retry_interval_max: Any,
    retries_before_backoff: Any,
    environment_type: Any,
    unrecoverable_error_should_crash: Any,
    local_matrix_servers: Any,
    blockchain_type: Any,
    contracts_path: Any,
    user_deposit_address: Any,
    logs_storage: Any,
    register_tokens: Any,
    start_raiden_apps: Any,
    routing_mode: Any,
    blockchain_query_interval: Any,
    resolver_ports: Any,
    enable_rest_api: Any,
    port_generator: Any,
    capabilities: Any,
) -> Generator[Any, None, None]: ...

@pytest.fixture
def resolvers(resolver_ports: Any) -> Generator[list[Any], None, None]: ...

@pytest.fixture
def adhoc_capability() -> bool: ...

@pytest.fixture
def capabilities(adhoc_capability: bool) -> CapabilitiesConfig: ...

@pytest.fixture
def pfs_mock(
    monkeypatch: Any,
    local_matrix_servers: Any,
    chain_id: Any,
    token_network_registry_address: Any,
    user_deposit_address: Any,
) -> PFSMock: ...

@pytest.fixture
def raiden_network(
    token_addresses: Any,
    token_network_registry_address: Any,
    one_to_n_address: Any,
    monitoring_service_address: Any,
    channels_per_node: Any,
    deposit: Any,
    settle_timeout: Any,
    chain_id: Any,
    blockchain_services: Any,
    reveal_timeout: Any,
    retry_interval_initial: Any,
    retry_interval_max: Any,
    retries_before_backoff: Any,
    environment_type: Any,
    unrecoverable_error_should_crash: Any,
    local_matrix_servers: Any,
    blockchain_type: Any,
    contracts_path: Any,
    user_deposit_address: Any,
    logs_storage: Any,
    register_tokens: Any,
    start_raiden_apps: Any,
    routing_mode: Any,
    blockchain_query_interval: Any,
    resolver_ports: Any,
    enable_rest_api: Any,
    port_generator: Any,
    capabilities: Any,
) -> Generator[Any, None, None]: ...

class RestartNode:
    async_result: Optional[AsyncResult[Any]]
    def __init__(self) -> None: ...
    def link_exception_to(self, result: AsyncResult[Any]) -> None: ...
    def __call__(self, service: RaidenService) -> None: ...

@pytest.fixture
def restart_node() -> RestartNode: ...
```