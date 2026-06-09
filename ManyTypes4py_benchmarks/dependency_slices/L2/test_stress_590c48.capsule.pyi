from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: to_canonical_address

# === Third-party dependency: flask ===
# Used symbols: url_for

# === Third-party dependency: gevent ===
# Used symbols: iwait, joinall, spawn

# === Third-party dependency: grequests ===
post: partial

# === Third-party dependency: pytest ===
# Used symbols: mark

# === Internal dependency: raiden.api.python ===
class RaidenAPI:
    def __init__(self, raiden: 'RaidenService') -> Any: ...

# === Internal dependency: raiden.api.rest ===
class APIServer(Runnable):
    def __init__(self, rest_api: 'RestAPI', config: RestApiConfig, eth_rpc_endpoint: str = ...) -> None: ...
    def start(self) -> None: ...
class RestAPI:
    def __init__(self, raiden_api: RaidenAPI = ..., rpc_client: JSONRPCClient = ...) -> None: ...

# === Internal dependency: raiden.constants ===
class RoutingMode(Enum): ...

# === Internal dependency: raiden.message_handler ===
class MessageHandler: ...

# === Internal dependency: raiden.network.transport ===
# re-export: from raiden.network.transport.matrix import MatrixTransport

# === Internal dependency: raiden.raiden_event_handler ===
class RaidenEventHandler(EventHandler):
    ...

# === Internal dependency: raiden.raiden_service ===
class RaidenService(Runnable):
    def __init__(self, rpc_client: JSONRPCClient, proxy_manager: ProxyManager, query_start_block: BlockNumber, raiden_bundle: RaidenBundle, services_bundle: Optional[ServicesBundle], transport: MatrixTransport, raiden_event_handler: EventHandler, message_handler: MessageHandler, routing_mode: RoutingMode, config: RaidenConfig, api_server: Optional[APIServer] = ..., pfs_proxy: PFSProxy = ...) -> None: ...

# === Internal dependency: raiden.settings ===
class RestApiConfig: ...

# === Internal dependency: raiden.tests.integration.api.utils ===
def wait_for_listening_port(port_number: Port, tries: int = ..., sleep: float = ..., pid: int = ...) -> None: ...

# === Internal dependency: raiden.tests.utils.detect_failure ===
def raise_on_failure(test_function: Callable) -> Callable: ...

# === Internal dependency: raiden.tests.utils.protocol ===
class HoldRaidenEventHandler(EventHandler):
    def __init__(self, wrapped_handler: EventHandler) -> Any: ...

# === Internal dependency: raiden.tests.utils.transfer ===
def watch_for_unlock_failures(*apps) -> Any: ...
def assert_synced_channel_state(token_network_address: TokenNetworkAddress, app0: RaidenService, balance0: Balance, pending_locks0: List[HashTimeLockState], app1: RaidenService, balance1: Balance, pending_locks1: List[HashTimeLockState]) -> None: ...
def wait_assert(func: Callable, *args, **kwargs) -> None: ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden: 'RaidenService') -> ChainState: ...
def get_token_network_address_by_token_address(chain_state: ChainState, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress) -> Optional[TokenNetworkAddress]: ...

# === Internal dependency: raiden.ui.startup ===
class RaidenBundle: ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address: AddressTypes) -> ChecksumAddress: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from typing import Tuple
# re-export: from eth_typing import Address
# re-export: from eth_typing import BlockNumber
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount
Host: NewType

# === Third-party dependency: structlog ===
# Used symbols: get_logger