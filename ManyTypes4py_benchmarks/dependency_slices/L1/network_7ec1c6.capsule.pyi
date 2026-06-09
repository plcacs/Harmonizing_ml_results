# === Third-party dependency: gevent ===
# Used symbols: joinall, sleep, spawn

# === Internal dependency: raiden.constants ===
GENESIS_BLOCK_NUMBER = BlockNumber(...)
BLOCK_ID_LATEST = 'latest'

# === Internal dependency: raiden.exceptions ===
class PFSReturnedError(ServiceRequestFailed): ...

# === Internal dependency: raiden.network.pathfinding ===
class PFSProxy:
    ...

# === Internal dependency: raiden.network.proxies.proxy_manager ===
class ProxyManagerMetadata: ...
class ProxyManager:
    def __init__(self, rpc_client, contract_manager, metadata): ...

# === Internal dependency: raiden.network.proxies.secret_registry ===
class SecretRegistry: ...

# === Internal dependency: raiden.network.proxies.service_registry ===
class ServiceRegistry: ...

# === Internal dependency: raiden.network.proxies.token_network_registry ===
class TokenNetworkRegistry: ...

# === Internal dependency: raiden.network.rpc.client ===
class JSONRPCClient:
    def __init__(self, web3, privkey, gas_price_strategy=..., block_num_confirmations=...): ...

# === Internal dependency: raiden.raiden_event_handler ===
class RaidenEventHandler(EventHandler):
    ...

# === Internal dependency: raiden.raiden_service ===
class RaidenService(Runnable):
    def __init__(self, rpc_client, proxy_manager, query_start_block, raiden_bundle, services_bundle, transport, raiden_event_handler, message_handler, routing_mode, config, api_server=..., pfs_proxy=...): ...

# === Internal dependency: raiden.settings ===
class MediationFeeConfig: ...
class MatrixTransportConfig:
    ...
class ServiceConfig: ...
class BlockchainConfig: ...
class RestApiConfig: ...
class RaidenConfig:
DEFAULT_RETRY_TIMEOUT = NetworkTimeout(...)
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS = BlockTimeout(...)

# === Internal dependency: raiden.tests.utils.app ===
def database_from_privatekey(base_dir, app_number): ...

# === Internal dependency: raiden.tests.utils.factories ===
UNIT_CHAIN_ID = ChainID(...)

# === Internal dependency: raiden.tests.utils.protocol ===
class WaitForMessage(MessageHandler):
    def __init__(self): ...
class HoldRaidenEventHandler(EventHandler):
    def __init__(self, wrapped_handler): ...

# === Internal dependency: raiden.tests.utils.transport ===
class TestMatrixTransport(MatrixTransport):
    def __init__(self, config, environment): ...

# === Internal dependency: raiden.transfer.identifiers ===
class CanonicalIdentifier:
    ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden): ...
def get_channelstate_by_token_network_and_partner(chain_state, token_network_address, partner_address): ...
def get_channelstate_by_canonical_identifier(chain_state, canonical_identifier): ...
def get_confirmed_blockhash(raiden): ...

# === Internal dependency: raiden.ui.app ===
def start_api_server(rpc_client, config, eth_rpc_endpoint): ...

# === Internal dependency: raiden.ui.startup ===
class RaidenBundle: ...
class ServicesBundle:
    ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address): ...
def to_hex_address(address): ...

# === Internal dependency: raiden.utils.typing ===
from typing import Tuple
from eth_typing import BlockNumber
from web3.types import BlockIdentifier
from raiden_contracts.utils.type_aliases import ChainID
T_BlockTimeout = int
BlockTimeout = NewType(...)
T_NetworkTimeout = float
NetworkTimeout = NewType(...)
Host = NewType(...)

# === Internal dependency: raiden.waiting ===
def wait_for_newchannel(raiden, token_network_registry_address, token_address, partner_address, retry_timeout): ...
def wait_for_participant_deposit(raiden, token_network_registry_address, token_address, partner_address, target_address, target_balance, retry_timeout): ...
def wait_for_token_network(raiden, token_network_registry_address, token_address, retry_timeout): ...

# === Third-party dependency: structlog ===
# Used symbols: get_logger