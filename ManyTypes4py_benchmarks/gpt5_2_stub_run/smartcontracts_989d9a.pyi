from typing import Any, List, Optional
from web3.contract import Contract
from raiden.network.proxies.monitoring_service import MonitoringService
from raiden.network.proxies.one_to_n import OneToN
from raiden.network.proxies.secret_registry import SecretRegistry
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.proxies.token import Token
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.proxies.token_network_registry import TokenNetworkRegistry
from raiden.network.proxies.user_deposit import UserDeposit
from raiden.utils.typing import (
    MonitoringServiceAddress,
    OneToNAddress,
    SecretRegistryAddress,
    ServiceRegistryAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    UserDepositAddress,
)

RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT: TokenAmount = ...
RED_EYES_PER_TOKEN_NETWORK_LIMIT: TokenAmount = ...

class ServicesSmartContracts:
    utility_token_proxy: Token
    utility_token_network_proxy: Optional[TokenNetwork]
    one_to_n_proxy: OneToN
    user_deposit_proxy: UserDeposit
    service_registry_proxy: ServiceRegistry
    monitoring_service: MonitoringService
    def __init__(
        self,
        utility_token_proxy: Token,
        utility_token_network_proxy: Optional[TokenNetwork],
        one_to_n_proxy: OneToN,
        user_deposit_proxy: UserDeposit,
        service_registry_proxy: ServiceRegistry,
        monitoring_service: MonitoringService,
    ) -> None: ...

class FixtureSmartContracts:
    secret_registry_proxy: SecretRegistry
    token_network_registry_proxy: TokenNetworkRegistry
    token_contracts: List[Contract]
    services_smart_contracts: Optional[ServicesSmartContracts]
    def __init__(
        self,
        secret_registry_proxy: SecretRegistry,
        token_network_registry_proxy: TokenNetworkRegistry,
        token_contracts: List[Contract],
        services_smart_contracts: Optional[ServicesSmartContracts],
    ) -> None: ...

def deploy_secret_registry(deploy_client: Any, contract_manager: Any, proxy_manager: Any) -> SecretRegistry: ...
def deploy_token_network_registry(
    secret_registry_deploy_result: Any,
    deploy_client: Any,
    contract_manager: Any,
    proxy_manager: Any,
    settle_timeout_min: Any,
    settle_timeout_max: Any,
    max_token_networks: Any,
) -> TokenNetworkRegistry: ...
def register_token(token_network_registry_deploy_result: Any, token_deploy_result: Any) -> TokenNetworkAddress: ...
def deploy_service_registry(token_deploy_result: Any, deploy_client: Any, contract_manager: Any, proxy_manager: Any) -> ServiceRegistry: ...
def deploy_one_to_n(
    user_deposit_deploy_result: Any,
    service_registry_deploy_result: Any,
    deploy_client: Any,
    contract_manager: Any,
    proxy_manager: Any,
    chain_id: Any,
) -> OneToN: ...
def deploy_monitoring_service(
    token_deploy_result: Any,
    user_deposit_deploy_result: Any,
    service_registry_deploy_result: Any,
    token_network_registry_deploy_result: Any,
    deploy_client: Any,
    contract_manager: Any,
    proxy_manager: Any,
) -> MonitoringService: ...
def deploy_user_deposit(token_deploy_result: Any, deploy_client: Any, contract_manager: Any, proxy_manager: Any) -> UserDeposit: ...
def transfer_user_deposit_tokens(user_deposit_deploy_result: Any, transfer_to: Any) -> None: ...
def fund_node(token_result: Any, proxy_manager: Any, to_address: Any, amount: Any) -> None: ...
def deploy_smart_contract_bundle_concurrently(
    deploy_client: Any,
    contract_manager: Any,
    proxy_manager: Any,
    chain_id: Any,
    environment_type: Any,
    max_token_networks: Any,
    number_of_tokens: Any,
    private_keys: Any,
    register_tokens: Any,
    settle_timeout_max: Any,
    settle_timeout_min: Any,
    token_amount: Any,
    token_contract_name: Any,
) -> FixtureSmartContracts: ...
def token_contract_name_fixture() -> str: ...
def max_token_networks_fixture() -> int: ...
def token_addresses_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> List[TokenAddress]: ...
def secret_registry_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> SecretRegistryAddress: ...
def service_registry_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[ServiceRegistryAddress]: ...
def user_deposit_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[UserDepositAddress]: ...
def one_to_n_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[OneToNAddress]: ...
def monitoring_service_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[MonitoringServiceAddress]: ...
def secret_registry_proxy_fixture(deploy_client: Any, secret_registry_address: SecretRegistryAddress, contract_manager: Any) -> SecretRegistry: ...
def token_network_registry_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> TokenNetworkRegistryAddress: ...
def token_network_proxy_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[TokenNetwork]: ...
def token_proxy_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts, environment_type: Any) -> Token: ...