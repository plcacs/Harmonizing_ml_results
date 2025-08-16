from dataclasses import dataclass
from raiden.constants import BLOCK_ID_LATEST, EMPTY_ADDRESS, SECONDS_PER_DAY, UINT256_MAX, Environment
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
from raiden.settings import MONITORING_REWARD
from raiden.utils.keys import privatekey_to_address
from raiden.utils.typing import Address, BlockNumber, Callable, ChainID, List, MonitoringServiceAddress, OneToNAddress, Optional, PrivateKey, SecretRegistryAddress, ServiceRegistryAddress, Set, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, UserDepositAddress
from raiden_contracts.constants import CONTRACT_CUSTOM_TOKEN, CONTRACT_MONITORING_SERVICE, CONTRACT_ONE_TO_N, CONTRACT_SECRET_REGISTRY, CONTRACT_SERVICE_REGISTRY, CONTRACT_TOKEN_NETWORK_REGISTRY, CONTRACT_USER_DEPOSIT
from raiden_contracts.contract_manager import ContractManager

@dataclass
class ServicesSmartContracts:
    utility_token_proxy: Token
    utility_token_network_proxy: TokenNetwork
    one_to_n_proxy: OneToN
    user_deposit_proxy: UserDeposit
    service_registry_proxy: ServiceRegistry
    monitoring_service: MonitoringService

@dataclass
class FixtureSmartContracts:
    secret_registry_proxy: SecretRegistry
    token_network_registry_proxy: TokenNetworkRegistry
    token_contracts: List[Token]
    services_smart_contracts: Optional[ServicesSmartContracts]

def deploy_secret_registry(deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: ProxyManager) -> SecretRegistry:
    ...

def deploy_token_network_registry(secret_registry_deploy_result: Callable[[], SecretRegistry], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: ProxyManager, settle_timeout_min: int, settle_timeout_max: int, max_token_networks: int) -> TokenNetworkRegistry:
    ...

def register_token(token_network_registry_deploy_result: Callable[[], TokenNetworkRegistry], token_deploy_result: Callable[[], Token]) -> TokenNetworkAddress:
    ...

def deploy_service_registry(token_deploy_result: Callable[[], Token], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: ProxyManager) -> ServiceRegistry:
    ...

def deploy_one_to_n(user_deposit_deploy_result: Callable[[], UserDeposit], service_registry_deploy_result: Callable[[], ServiceRegistry], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: ProxyManager, chain_id: ChainID) -> OneToN:
    ...

def deploy_monitoring_service(token_deploy_result: Callable[[], Token], user_deposit_deploy_result: Callable[[], UserDeposit], service_registry_deploy_result: Callable[[], ServiceRegistry], token_network_registry_deploy_result: Callable[[], TokenNetworkRegistry], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: ProxyManager) -> MonitoringService:
    ...

def deploy_user_deposit(token_deploy_result: Callable[[], Token], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: ProxyManager) -> UserDeposit:
    ...

def transfer_user_deposit_tokens(user_deposit_deploy_result: Callable[[], UserDeposit], transfer_to: Address) -> None:
    ...

def fund_node(token_result: Callable[[], Token], proxy_manager: ProxyManager, to_address: Address, amount: TokenAmount) -> None:
    ...

def deploy_smart_contract_bundle_concurrently(deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: ProxyManager, chain_id: ChainID, environment_type: Environment, max_token_networks: int, number_of_tokens: int, private_keys: List[PrivateKey], register_tokens: bool, settle_timeout_max: int, settle_timeout_min: int, token_amount: TokenAmount, token_contract_name: str) -> FixtureSmartContracts:
    ...

@pytest.fixture(name='token_contract_name')
def token_contract_name_fixture() -> str:
    ...

@pytest.fixture(name='max_token_networks')
def max_token_networks_fixture() -> int:
    ...

@pytest.fixture(name='token_addresses')
def token_addresses_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> List[TokenAddress]:
    ...

@pytest.fixture(name='secret_registry_address')
def secret_registry_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> SecretRegistryAddress:
    ...

@pytest.fixture(name='service_registry_address')
def service_registry_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[ServiceRegistryAddress]:
    ...

@pytest.fixture(name='user_deposit_address')
def user_deposit_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[UserDepositAddress]:
    ...

@pytest.fixture(name='one_to_n_address')
def one_to_n_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[OneToNAddress]:
    ...

@pytest.fixture(name='monitoring_service_address')
def monitoring_service_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[MonitoringServiceAddress]:
    ...

@pytest.fixture(name='secret_registry_proxy')
def secret_registry_proxy_fixture(deploy_client: JSONRPCClient, secret_registry_address: SecretRegistryAddress, contract_manager: ContractManager) -> SecretRegistry:
    ...

@pytest.fixture(name='token_network_registry_address')
def token_network_registry_address_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> TokenNetworkRegistryAddress:
    ...

@pytest.fixture(name='token_network_proxy')
def token_network_proxy_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[TokenNetwork]:
    ...

@pytest.fixture(name='token_proxy')
def token_proxy_fixture(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts, environment_type: Environment) -> Token:
    ...
