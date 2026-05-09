from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set, Union
from eth_utils import Address, BlockNumber
from gevent import Greenlet
from web3.contract import Contract
from raiden.settings import MONITORING_REWARD
from raiden.utils.typing import (
    BlockNumber,
    Callable,
    ChainID,
    List,
    MonitoringServiceAddress,
    OneToNAddress,
    Optional,
    PrivateKey,
    SecretRegistryAddress,
    ServiceRegistryAddress,
    Set,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    UserDepositAddress,
)
from raiden_contracts.contract_manager import ContractManager
from raiden.network.proxies.secret_registry import SecretRegistry
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.proxies.token import Token
from raiden.network.proxies.token_network_registry import TokenNetworkRegistry
from raiden.network.proxies.one_to_n import OneToN
from raiden.network.proxies.user_deposit import UserDeposit
from raiden.network.proxies.monitoring_service import MonitoringService
from raiden.network.rpc.client import JSONRPCClient

RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT = TokenAmount(int(0.075 * 10 ** 18))
RED_EYES_PER_TOKEN_NETWORK_LIMIT = TokenAmount(int(250 * 10 ** 18))

@dataclass
class ServicesSmartContracts:
    pass

@dataclass
class FixtureSmartContracts:
    pass

def deploy_secret_registry(deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: Any) -> SecretRegistry:
    ...

def deploy_token_network_registry(secret_registry_deploy_result: Callable[[], SecretRegistry], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: Any, settle_timeout_min: int, settle_timeout_max: int, max_token_networks: int) -> TokenNetworkRegistry:
    ...

def register_token(token_network_registry_deploy_result: Callable[[], TokenNetworkRegistry], token_deploy_result: Callable[[], Contract]) -> TokenNetworkAddress:
    ...

def deploy_service_registry(token_deploy_result: Callable[[], Contract], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: Any) -> ServiceRegistry:
    ...

def deploy_one_to_n(user_deposit_deploy_result: Callable[[], UserDeposit], service_registry_deploy_result: Callable[[], ServiceRegistry], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: Any, chain_id: ChainID) -> OneToN:
    ...

def deploy_monitoring_service(token_deploy_result: Callable[[], Contract], user_deposit_deploy_result: Callable[[], UserDeposit], service_registry_deploy_result: Callable[[], ServiceRegistry], token_network_registry_deploy_result: Callable[[], TokenNetworkRegistry], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: Any) -> MonitoringService:
    ...

def deploy_user_deposit(token_deploy_result: Callable[[], Contract], deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: Any) -> UserDeposit:
    ...

def transfer_user_deposit_tokens(user_deposit_deploy_result: Callable[[], UserDeposit], transfer_to: Address) -> None:
    ...

def fund_node(token_result: Callable[[], Contract], proxy_manager: Any, to_address: Address, amount: TokenAmount) -> None:
    ...

@pytest.fixture
def deploy_smart_contract_bundle_concurrently(deploy_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: Any, chain_id: ChainID, environment_type: str, max_token_networks: int, number_of_tokens: int, private_keys: List[PrivateKey], register_tokens: bool, settle_timeout_max: int, settle_timeout_min: int, token_amount: TokenAmount, token_contract_name: str) -> FixtureSmartContracts:
    ...

@pytest.fixture
def token_contract_name() -> str:
    ...

@pytest.fixture
def max_token_networks() -> int:
    ...

@pytest.fixture
def token_addresses(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> List[TokenAddress]:
    ...

@pytest.fixture
def secret_registry_address(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Address:
    ...

@pytest.fixture
def service_registry_address(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[Address]:
    ...

@pytest.fixture
def user_deposit_address(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[Address]:
    ...

@pytest.fixture
def one_to_n_address(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[Address]:
    ...

@pytest.fixture
def monitoring_service_address(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[Address]:
    ...

@pytest.fixture
def secret_registry_proxy(deploy_client: JSONRPCClient, secret_registry_address: Optional[Address], contract_manager: ContractManager) -> SecretRegistry:
    ...

@pytest.fixture
def token_network_registry_address(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Address:
    ...

@pytest.fixture
def token_network_proxy(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts) -> Optional[Any]:
    ...

@pytest.fixture
def token_proxy(deploy_smart_contract_bundle_concurrently: FixtureSmartContracts, environment_type: str) -> Token:
    ...