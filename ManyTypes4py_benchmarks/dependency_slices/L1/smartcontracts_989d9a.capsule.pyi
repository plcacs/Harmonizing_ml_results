# === Third-party dependency: eth_utils ===
# Used symbols: to_canonical_address

# === Third-party dependency: gevent ===
# Used symbols: Greenlet, joinall, spawn

# === Third-party dependency: pytest ===
# Used symbols: fixture

# === Internal dependency: raiden.constants ===
class Environment(Enum): ...
UINT256_MAX = 2 ** 256 - 1
SECONDS_PER_DAY = 24 * 60 * 60
EMPTY_ADDRESS = b'\x00' * 20
BLOCK_ID_LATEST = 'latest'

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
    def __init__(self, jsonrpc_client, token_address, contract_manager, block_identifier): ...

# === Internal dependency: raiden.network.proxies.token_network ===
class TokenNetwork: ...

# === Internal dependency: raiden.network.proxies.token_network_registry ===
class TokenNetworkRegistry: ...

# === Internal dependency: raiden.network.proxies.user_deposit ===
class UserDeposit: ...

# === Internal dependency: raiden.settings ===
MONITORING_REWARD = TokenAmount(...)

# === Internal dependency: raiden.tests.utils.smartcontracts ===
def deploy_token(deploy_client, contract_manager, initial_amount, decimals, token_name, token_symbol, token_contract_name): ...

# === Internal dependency: raiden.utils.keys ===
def privatekey_to_address(private_key_bin): ...

# === Internal dependency: raiden.utils.typing ===
from eth_typing import BlockNumber
from raiden_contracts.utils.type_aliases import TokenAmount
T_TokenNetworkRegistryAddress = bytes
TokenNetworkRegistryAddress = NewType(...)
T_TokenAddress = bytes
TokenAddress = NewType(...)
T_UserDepositAddress = bytes
UserDepositAddress = NewType(...)
T_MonitoringServiceAddress = bytes
MonitoringServiceAddress = NewType(...)
T_ServiceRegistryAddress = bytes
ServiceRegistryAddress = NewType(...)
T_OneToNAddress = bytes
OneToNAddress = NewType(...)
T_TokenNetworkAddress = bytes
TokenNetworkAddress = NewType(...)
T_SecretRegistryAddress = bytes
SecretRegistryAddress = NewType(...)

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