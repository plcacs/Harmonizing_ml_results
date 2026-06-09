from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: to_canonical_address

# === Third-party dependency: gevent ===
# Used symbols: Greenlet, joinall, spawn

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Internal dependency: raiden.constants ===
class Environment(Enum): ...
BLOCK_ID_LATEST: Literal['latest']
UINT256_MAX: Any
SECONDS_PER_DAY: Any
EMPTY_ADDRESS: Any

# === Internal dependency: raiden.network.proxies.monitoring_service ===
class MonitoringService: ...

# === Internal dependency: raiden.network.proxies.one_to_n ===
class OneToN: ...

# === Internal dependency: raiden.network.proxies.secret_registry ===
class SecretRegistry: ...

# === Internal dependency: raiden.network.proxies.service_registry ===
class ServiceRegistry: ...

# === Internal dependency: raiden.network.proxies.token ===
class Token:
    def __init__(self, jsonrpc_client: JSONRPCClient, token_address: TokenAddress, contract_manager: ContractManager, block_identifier: BlockIdentifier) -> None: ...

# === Internal dependency: raiden.network.proxies.token_network ===
class TokenNetwork: ...

# === Internal dependency: raiden.network.proxies.token_network_registry ===
class TokenNetworkRegistry: ...

# === Internal dependency: raiden.network.proxies.user_deposit ===
class UserDeposit: ...

# === Internal dependency: raiden.settings ===
MONITORING_REWARD: TokenAmount

# === Internal dependency: raiden.tests.utils.smartcontracts ===
def deploy_token(deploy_client: JSONRPCClient, contract_manager: ContractManager, initial_amount: TokenAmount, decimals: int, token_name: str, token_symbol: str, token_contract_name: str) -> Contract: ...

# === Internal dependency: raiden.utils.keys ===
def privatekey_to_address(private_key_bin: bytes) -> Address: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from eth_typing import BlockNumber
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount
TokenNetworkRegistryAddress: NewType
TokenAddress: NewType
UserDepositAddress: NewType
MonitoringServiceAddress: NewType
ServiceRegistryAddress: NewType
OneToNAddress: NewType
TokenNetworkAddress: NewType
SecretRegistryAddress: NewType

# === Third-party dependency: raiden_contracts.constants ===
CONTRACT_TOKEN_NETWORK_REGISTRY: str
CONTRACT_SECRET_REGISTRY: str
CONTRACT_CUSTOM_TOKEN: str
CONTRACT_MONITORING_SERVICE: str
CONTRACT_SERVICE_REGISTRY: str
CONTRACT_USER_DEPOSIT: str
CONTRACT_ONE_TO_N: str

# === Unresolved dependency: web3.contract ===
# Used unresolved symbols: Contract