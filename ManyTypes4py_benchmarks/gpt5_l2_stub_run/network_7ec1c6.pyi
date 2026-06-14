from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union
from web3 import Web3

from raiden.constants import Environment, RoutingMode
from raiden.exceptions import PFSReturnedError
from raiden.network.pathfinding import PFSProxy
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.proxies.secret_registry import SecretRegistry
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.proxies.token_network_registry import TokenNetworkRegistry
from raiden.network.rpc.client import JSONRPCClient
from raiden.raiden_event_handler import RaidenEventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import (
    DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS,
    DEFAULT_RETRY_TIMEOUT,
    BlockchainConfig,
    CapabilitiesConfig,
    MatrixTransportConfig,
    MediationFeeConfig,
    RaidenConfig,
    RestApiConfig,
    ServiceConfig,
)
from raiden.tests.utils.transport import ParsedURL, TestMatrixTransport
from raiden.ui.startup import RaidenBundle, ServicesBundle
from raiden.utils.typing import (
    Address,
    BlockNumber,
    BlockTimeout,
    ChainID,
    MonitoringServiceAddress,
    NetworkTimeout,
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

AppChannels = Iterable[Tuple[RaidenService, RaidenService]]

log: Any = ...
CHAIN: object = ...


@dataclass
class BlockchainServices:
    deploy_registry: TokenNetworkRegistry
    secret_registry: SecretRegistry
    service_registry: Optional[ServiceRegistry]
    proxy_manager: ProxyManager
    blockchain_services: List[ProxyManager]


def check_channel(
    app1: RaidenService,
    app2: RaidenService,
    token_network_address: TokenNetworkAddress,
    settle_timeout: BlockTimeout,
    deposit_amount: TokenAmount,
) -> None: ...


def payment_channel_open_and_deposit(
    app0: RaidenService,
    app1: RaidenService,
    token_address: TokenAddress,
    deposit: TokenAmount,
    settle_timeout: BlockTimeout,
) -> None: ...


def create_all_channels_for_network(
    app_channels: AppChannels,
    token_addresses: Iterable[TokenAddress],
    channel_individual_deposit: TokenAmount,
    channel_settle_timeout: BlockTimeout,
) -> None: ...


def network_with_minimum_channels(
    apps: List[RaidenService],
    channels_per_node: int,
) -> Iterator[Tuple[RaidenService, RaidenService]]: ...


def create_network_channels(
    raiden_apps: List[RaidenService],
    channels_per_node: Union[int, object],
) -> List[Tuple[RaidenService, RaidenService]]: ...


def create_sequential_channels(
    raiden_apps: List[RaidenService],
    channels_per_node: Union[int, object],
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
    reveal_timeout: BlockTimeout,
    settle_timeout: BlockTimeout,
    database_basedir: Path,
    retry_interval_initial: float,
    retry_interval_max: float,
    retries_before_backoff: int,
    environment_type: Environment,
    unrecoverable_error_should_crash: bool,
    local_matrix_url: Optional[ParsedURL],
    routing_mode: RoutingMode,
    blockchain_query_interval: float,
    resolver_ports: List[Optional[Port]],
    enable_rest_api: bool,
    port_generator: Iterator[Port],
    capabilities_config: CapabilitiesConfig,
) -> List[RaidenService]: ...


class SimplePFSProxy(PFSProxy):
    def __init__(self, services: Iterable[RaidenService]) -> None: ...
    def query_address_metadata(self, address: Address) -> Dict[str, Any]: ...
    def set_services(self, services: Iterable[RaidenService]) -> None: ...


def parallel_start_apps(raiden_apps: Iterable[RaidenService]) -> None: ...


def jsonrpc_services(
    proxy_manager: ProxyManager,
    private_keys: Iterable[PrivateKey],
    secret_registry_address: SecretRegistryAddress,
    service_registry_address: Optional[ServiceRegistryAddress],
    token_network_registry_address: TokenNetworkRegistryAddress,
    web3: Web3,
    contract_manager: ContractManager,
) -> BlockchainServices: ...


def wait_for_alarm_start(
    raiden_apps: Iterable[RaidenService],
    retry_timeout: NetworkTimeout = ...,
) -> None: ...


def wait_for_usable_channel(
    raiden: RaidenService,
    partner_address: Address,
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_address: TokenAddress,
    our_deposit: TokenAmount,
    partner_deposit: TokenAmount,
    retry_timeout: NetworkTimeout = ...,
) -> None: ...


def wait_for_token_networks(
    raiden_apps: Iterable[RaidenService],
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_addresses: Iterable[TokenAddress],
    retry_timeout: NetworkTimeout = ...,
) -> None: ...


def wait_for_channels(
    app_channels: AppChannels,
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_addresses: Iterable[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: NetworkTimeout = ...,
) -> None: ...