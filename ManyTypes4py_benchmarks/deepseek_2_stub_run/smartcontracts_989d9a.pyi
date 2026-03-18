```python
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set
from eth_utils import to_canonical_address
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
    utility_token_proxy: Any = ...
    utility_token_network_proxy: Any = ...
    one_to_n_proxy: Any = ...
    user_deposit_proxy: Any = ...
    service_registry_proxy: Any = ...
    monitoring_service: Any = ...

@dataclass
class FixtureSmartContracts:
    secret_registry_proxy: Any = ...
    token_network_registry_proxy: Any = ...
    token_contracts: Any = ...
    services_smart_contracts: Any = ...

def deploy_secret_registry(
    deploy_client: Any,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> Any: ...

def deploy_token_network_registry(
    secret_registry_deploy_result: Callable[[], Any],
    deploy_client: Any,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
    settle_timeout_min: Any,
    settle_timeout_max: Any,
    max_token_networks: Any,
) -> Any: ...

def register_token(
    token_network_registry_deploy_result: Callable[[], Any],
    token_deploy_result: Callable[[], Any],
) -> Any: ...

def deploy_service_registry(
    token_deploy_result: Callable[[], Any],
    deploy_client: Any,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> Any: ...

def deploy_one_to_n(
    user_deposit_deploy_result: Callable[[], Any],
    service_registry_deploy_result: Callable[[], Any],
    deploy_client: Any,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
    chain_id: ChainID,
) -> Any: ...

def deploy_monitoring_service(
    token_deploy_result: Callable[[], Any],
    user_deposit_deploy_result: Callable[[], Any],
    service_registry_deploy_result: Callable[[], Any],
    token_network_registry_deploy_result: Callable[[], Any],
    deploy_client: Any,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> Any: ...

def deploy_user_deposit(
    token_deploy_result: Callable[[], Any],
    deploy_client: Any,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
) -> Any: ...

def transfer_user_deposit_tokens(
    user_deposit_deploy_result: Callable[[], Any],
    transfer_to: Any,
) -> None: ...

def fund_node(
    token_result: Callable[[], Any],
    proxy_manager: ProxyManager,
    to_address: Address,
    amount: TokenAmount,
) -> None: ...

def deploy_smart_contract_bundle_concurrently(
    deploy_client: Any,
    contract_manager: ContractManager,
    proxy_manager: ProxyManager,
    chain_id: ChainID,
    environment_type: Environment,
    max_token_networks: Any,
    number_of_tokens: Any,
    private_keys: List[PrivateKey],
    register_tokens: Any,
    settle_timeout_max: Any,
    settle_timeout_min: Any,
    token_amount: TokenAmount,
    token_contract_name: Any,
) -> FixtureSmartContracts: ...

def token_contract_name_fixture() -> Any: ...

def max_token_networks_fixture() -> Any: ...

def token_addresses_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> List[TokenAddress]: ...

def secret_registry_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Any: ...

def service_registry_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[Any]: ...

def user_deposit_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[Any]: ...

def one_to_n_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[Any]: ...

def monitoring_service_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[Any]: ...

def secret_registry_proxy_fixture(
    deploy_client: JSONRPCClient,
    secret_registry_address: SecretRegistryAddress,
    contract_manager: ContractManager,
) -> SecretRegistry: ...

def token_network_registry_address_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Any: ...

def token_network_proxy_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
) -> Optional[TokenNetwork]: ...

def token_proxy_fixture(
    deploy_smart_contract_bundle_concurrently: FixtureSmartContracts,
    environment_type: Environment,
) -> Token: ...
```