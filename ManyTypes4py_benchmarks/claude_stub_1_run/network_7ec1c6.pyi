```pyi
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Set, Tuple

from raiden.network.pathfinding import PFSProxy
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.raiden_service import RaidenService
from raiden.settings import CapabilitiesConfig
from raiden.ui.startup import ServicesBundle
from raiden.utils.typing import (
    Address,
    BlockNumber,
    ChainID,
    Host,
    MonitoringServiceAddress,
    OneToNAddress,
    Port,
    PrivateKey,
    SecretRegistryAddress,
    ServiceRegistryAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    UserDepositAddress,
)
from raiden_contracts.contract_manager import ContractManager
from web3 import Web3

AppChannels = Iterable[Tuple[RaidenService, RaidenService]]

CHAIN: object

@dataclass
class BlockchainServices:
    deploy_registry: Any
    secret_registry: Any
    service_registry: Optional[Any]
    proxy_manager: ProxyManager
    blockchain_services: List[ProxyManager]

def check_channel(
    app1: RaidenService,
    app2: RaidenService,
    token_network_address: TokenNetworkAddress,
    settle_timeout: int,
    deposit_amount: TokenAmount,
) -> None: ...

def payment_channel_open_and_deposit(
    app0: RaidenService,
    app1: RaidenService,
    token_address: TokenAddress,
    deposit: TokenAmount,
    settle_timeout: int,
) -> None: ...

def create_all_channels_for_network(
    app_channels: AppChannels,
    token_addresses: Iterable[TokenAddress],
    channel_individual_deposit: TokenAmount,
    channel_settle_timeout: int,
) -> None: ...

def network_with_minimum_channels(
    apps: Iterable[RaidenService],
    channels_per_node: int,
) -> Iterator[Tuple[RaidenService, RaidenService]]: ...

def create_network_channels(
    raiden_apps: List[RaidenService],
    channels_per_node: Any,
) -> List[Tuple[RaidenService, RaidenService]]: ...

def create_sequential_channels(
    raiden_apps: List[RaidenService],
    channels_per_node: Any,
) -> List[Tuple[RaidenService, RaidenService]]: ...

def create_apps(
    chain_id: ChainID,
    contracts_path: Path,
    blockchain_services: Iterable[ProxyManager],
    token_network_registry_address: TokenNetworkRegistryAddress,
    one_to_n_address: Optional[OneToNAddress],
    secret_registry_address: SecretRegistryAddress,
    service_registry_address: Optional[ServiceRegistryAddress],
    user_deposit_address: Optional[UserDepositAddress],
    monitoring_service_contract_address: Optional[MonitoringServiceAddress],
    reveal_timeout: int,
    settle_timeout: int,
    database_basedir: Path,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
    environment_type: Any,
    unrecoverable_error_should_crash: bool,
    local_matrix_url: Optional[str],
    routing_mode: Any,
    blockchain_query_interval: float,
    resolver_ports: List[Optional[Port]],
    enable_rest_api: bool,
    port_generator: Any,
    capabilities_config: CapabilitiesConfig,
) -> List[RaidenService]: ...

class SimplePFSProxy(PFSProxy):
    def __init__(self, services: List[RaidenService]) -> None: ...
    def query_address_metadata(self, address: Address) -> Any: ...
    def set_services(self, services: List[RaidenService]) -> None: ...

def parallel_start_apps(raiden_apps: List[RaidenService]) -> None: ...

def jsonrpc_services(
    proxy_manager: ProxyManager,
    private_keys: List[PrivateKey],
    secret_registry_address: SecretRegistryAddress,
    service_registry_address: Optional[ServiceRegistryAddress],
    token_network_registry_address: TokenNetworkRegistryAddress,
    web3: Web3,
    contract_manager: ContractManager,
) -> BlockchainServices: ...

def wait_for_alarm_start(
    raiden_apps: Iterable[RaidenService],
    retry_timeout: float = ...,
) -> None: ...

def wait_for_usable_channel(
    raiden: RaidenService,
    partner_address: Address,
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_address: TokenAddress,
    our_deposit: TokenAmount,
    partner_deposit: TokenAmount,
    retry_timeout: float = ...,
) -> None: ...

def wait_for_token_networks(
    raiden_apps: Iterable[RaidenService],
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_addresses: Iterable[TokenAddress],
    retry_timeout: float = ...,
) -> None: ...

def wait_for_channels(
    app_channels: AppChannels,
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_addresses: Iterable[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: float = ...,
) -> None: ...
```