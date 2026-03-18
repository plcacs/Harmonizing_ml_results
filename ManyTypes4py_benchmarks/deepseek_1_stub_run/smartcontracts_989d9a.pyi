```python
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set
from eth_utils.address import ChecksumAddress
from gevent import Greenlet
from web3.contract import Contract
from raiden.constants import Environment
from raiden.network.proxies.monitoring_service import MonitoringService
from raiden.network.proxies.one_to_n import OneToN
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.network.proxies.secret_registry import SecretRegistry
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.proxies.token import Token
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.proxies.token_network_registry import TokenNetworkRegistry
from raiden.network.proxies.user_deposit import UserDeposit
from raiden.network.rpc.client import JSONRPCClient
from raiden.utils.typing import (
    Address,
    BlockNumber,
    ChainID,
    MonitoringServiceAddress,
    OneToNAddress,
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

RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT: TokenAmount = ...
RED_EYES_PER_TOKEN_NETWORK_LIMIT: TokenAmount = ...

@dataclass
class ServicesSmartContracts:
    utility_token_proxy: Optional[Token] = ...
    utility_token_network_proxy: Optional[TokenNetwork] = ...
    one_to_n_proxy: Optional[OneToN] = ...
    user_deposit_proxy: Optional[UserDeposit] = ...
    service_registry_proxy: Optional[ServiceRegistry] = ...
    monitoring_service: Optional[MonitoringService] = ...

@dataclass
class FixtureSmartContracts:
    secret_registry_proxy: SecretRegistry = ...
    token_network_registry_proxy: TokenNetworkRegistry = ...
    token_contracts: List[Contract] = ...
    services_smart_contracts: Optional[ServicesSmartContracts] = ...

def deploy_secret_registry(
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> SecretRegistry: ...

def deploy_token_network_registry(
    secret_registry_deploy_result: Callable[[], SecretRegistry],
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
    settle_timeout_min: int,
    settle_timeout_max: int,
    max_token_networks: int,
) -> TokenNetworkRegistry: ...

def register_token(
    token_network_registry_deploy_result: Callable[[], TokenNetworkRegistry],
    token_deploy_result: Callable[[], Contract],
) -> TokenNetworkAddress: ...

def deploy_service_registry(
    token_deploy_result: Callable[[], Contract],
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> ServiceRegistry: ...

def deploy_one_to_n(
    user_deposit_deploy_result: Callable[[], UserDeposit],
    service_registry_deploy_result: Callable[[], ServiceRegistry],
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
    chain_id: ChainID,
) -> OneToN: ...

def deploy_monitoring_service(
    token_deploy_result: Callable[[], Contract],
    user_deposit_deploy_result: Callable[[], UserDeposit],
    service_registry_deploy_result: Callable[[], ServiceRegistry],
    token_network_registry_deploy_result: Callable[[], TokenNetworkRegistry],
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> MonitoringService: ...

def deploy_user_deposit(
    token_deploy_result: Callable[[], Contract],
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> UserDeposit: ...

def transfer_user_deposit_tokens(
    user_deposit_deploy_result: Callable[[], UserDeposit],
    transfer_to: Address,
) -> None: ...

def fund_node(
    token_result: Callable[[], Contract],
    proxy_manager: ProxyManager,
    to_address: Address,
    amount: TokenAmount,
) -> None: ...

def deploy_smart_contract_bundle_concurrently(
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
    chain_id: ChainID,
    environment_type: Environment,
    max_token_networks: int,
    number_of_tokens: int,
    private_keys: List[PrivateKey],
    register_tokens: bool,
    settle_timeout_max: int,
    settle_timeout_min: int,
    token_amount: TokenAmount,
    token_contract_name: str,
) -> FixtureSmartContracts: ...

def token_contract_name_fixture() -> str: ...

def max_token_networks_fixture() -> int: ...

def token_addresses_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> List[TokenAddress]: ...

def secret_registry_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> SecretRegistryAddress: ...

def service_registry_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[ServiceRegistryAddress]: ...

def user_deposit_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[UserDepositAddress]: ...

def one_to_n_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[OneToNAddress]: ...

def monitoring_service_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[MonitoringServiceAddress]: ...

def secret_registry_proxy_fixture(
    deploy_client: JSONRPCClient,
    secret_registry_address: SecretRegistryAddress,
    contract_manager: ContractManager,
) -> SecretRegistry: ...

def token_network_registry_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> TokenNetworkRegistryAddress: ...

def token_network_proxy_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[TokenNetwork]: ...

def token_proxy_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
    environment_type: Environment,
) -> Token: ...
```