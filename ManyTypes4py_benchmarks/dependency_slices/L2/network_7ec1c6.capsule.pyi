from typing import Any

# === Third-party dependency: gevent ===
# Used symbols: joinall, sleep, spawn

# === Internal dependency: raiden.constants ===
BLOCK_ID_LATEST: Literal['latest']
GENESIS_BLOCK_NUMBER: BlockNumber

# === Internal dependency: raiden.exceptions ===
class PFSReturnedError(ServiceRequestFailed): ...

# === Internal dependency: raiden.network.pathfinding ===
class PFSProxy:
    ...

# === Internal dependency: raiden.network.proxies.proxy_manager ===
class ProxyManagerMetadata: ...
class ProxyManager:
    def __init__(self, rpc_client: JSONRPCClient, contract_manager: ContractManager, metadata: ProxyManagerMetadata) -> None: ...

# === Internal dependency: raiden.network.proxies.secret_registry ===
class SecretRegistry: ...

# === Internal dependency: raiden.network.proxies.service_registry ===
class ServiceRegistry: ...

# === Internal dependency: raiden.network.proxies.token_network_registry ===
class TokenNetworkRegistry: ...

# === Internal dependency: raiden.network.rpc.client ===
class JSONRPCClient:
    def __init__(self, web3: Web3, privkey: PrivateKey, gas_price_strategy: Callable = ..., block_num_confirmations: int = ...) -> None: ...

# === Internal dependency: raiden.raiden_event_handler ===
class RaidenEventHandler(EventHandler):
    ...

# === Internal dependency: raiden.raiden_service ===
class RaidenService(Runnable):
    def __init__(self, rpc_client: JSONRPCClient, proxy_manager: ProxyManager, query_start_block: BlockNumber, raiden_bundle: RaidenBundle, services_bundle: Optional[ServicesBundle], transport: MatrixTransport, raiden_event_handler: EventHandler, message_handler: MessageHandler, routing_mode: RoutingMode, config: RaidenConfig, api_server: Optional[APIServer] = ..., pfs_proxy: PFSProxy = ...) -> None: ...

# === Internal dependency: raiden.settings ===
class MediationFeeConfig: ...
class MatrixTransportConfig:
    ...
class ServiceConfig: ...
class BlockchainConfig: ...
class RestApiConfig: ...
class RaidenConfig:
DEFAULT_RETRY_TIMEOUT: NetworkTimeout
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS: BlockTimeout

# === Internal dependency: raiden.tests.utils.app ===
def database_from_privatekey(base_dir, app_number) -> Any: ...

# === Internal dependency: raiden.tests.utils.factories ===
UNIT_CHAIN_ID: ChainID

# === Internal dependency: raiden.tests.utils.protocol ===
class WaitForMessage(MessageHandler):
    def __init__(self) -> Any: ...
class HoldRaidenEventHandler(EventHandler):
    def __init__(self, wrapped_handler: EventHandler) -> Any: ...

# === Internal dependency: raiden.tests.utils.transport ===
class TestMatrixTransport(MatrixTransport):
    def __init__(self, config: MatrixTransportConfig, environment: Environment) -> None: ...

# === Internal dependency: raiden.transfer.identifiers ===
class CanonicalIdentifier:
    ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden: 'RaidenService') -> ChainState: ...
def get_channelstate_by_token_network_and_partner(chain_state: ChainState, token_network_address: TokenNetworkAddress, partner_address: Address) -> Optional[NettingChannelState]: ...
def get_channelstate_by_canonical_identifier(chain_state: ChainState, canonical_identifier: CanonicalIdentifier) -> Optional[NettingChannelState]: ...
def get_confirmed_blockhash(raiden: 'RaidenService') -> BlockHash: ...

# === Internal dependency: raiden.ui.app ===
def start_api_server(rpc_client: JSONRPCClient, config: RestApiConfig, eth_rpc_endpoint: str) -> APIServer: ...

# === Internal dependency: raiden.ui.startup ===
class RaidenBundle: ...
class ServicesBundle:
    ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address: AddressTypes) -> ChecksumAddress: ...
def to_hex_address(address: AddressTypes) -> AddressHex: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from typing import Tuple
# re-export: from eth_typing import BlockNumber
# re-export: from web3.types import BlockIdentifier
# re-export: from raiden_contracts.utils.type_aliases import ChainID
BlockTimeout: NewType
NetworkTimeout: NewType
Host: NewType

# === Internal dependency: raiden.waiting ===
def wait_for_newchannel(raiden: 'RaidenService', token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address, retry_timeout: float) -> None: ...
def wait_for_participant_deposit(raiden: 'RaidenService', token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address, target_address: Address, target_balance: TokenAmount, retry_timeout: float) -> None: ...
def wait_for_token_network(raiden: 'RaidenService', token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, retry_timeout: float) -> None: ...

# === Third-party dependency: structlog ===
# Used symbols: get_logger