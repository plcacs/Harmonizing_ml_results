```pyi
from pathlib import Path
from typing import Any, Callable, Optional, TextIO, Tuple
from eth_typing import URI
from raiden.accounts import AccountManager
from raiden.api.rest import APIServer
from raiden.constants import ChainID, Environment
from raiden.message_handler import MessageHandler
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.network.rpc.client import JSONRPCClient
from raiden.network.transport import MatrixTransport
from raiden.raiden_event_handler import EventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import MatrixTransportConfig, RaidenConfig
from raiden.utils.typing import Address, BlockTimeout, Endpoint, PrivateKey
from structlog.testing import LogCapture

def fetch_available_matrix_servers(transport_config: MatrixTransportConfig, environment_type: Environment) -> None: ...
def get_account_and_private_key(account_manager: AccountManager, address: Optional[str], password_file: Optional[str]) -> Tuple[Address, PrivateKey]: ...
def get_smart_contracts_start_at(chain_id: ChainID) -> int: ...
def get_min_reveal_timeout(chain_id: ChainID) -> BlockTimeout: ...
def rpc_normalized_endpoint(eth_rpc_endpoint: str) -> URI: ...
def start_api_server(rpc_client: JSONRPCClient, config: Any, eth_rpc_endpoint: str) -> APIServer: ...
def setup_raiden_config(
    eth_rpc_endpoint: str,
    api_address: str,
    rpc: bool,
    rpccorsdomain: Optional[str],
    console: bool,
    web_ui: bool,
    matrix_server: Optional[str],
    chain_id: ChainID,
    environment_type: Environment,
    development_environment: bool,
    unrecoverable_error_should_crash: bool,
    pathfinding_max_paths: int,
    enable_monitoring: bool,
    resolver_endpoint: Optional[str],
    default_reveal_timeout: int,
    default_settle_timeout: int,
    flat_fee: Any,
    proportional_fee: Any,
    proportional_imbalance_fee: Any,
    blockchain_query_interval: float,
    cap_mediation_fees: bool,
    enable_tracing: bool,
    **kwargs: Any
) -> RaidenConfig: ...
def run_raiden_service(
    config: RaidenConfig,
    eth_rpc_endpoint: str,
    address: Optional[str],
    keystore_path: str,
    gas_price: Any,
    user_deposit_contract_address: Optional[str],
    sync_check: bool,
    password_file: Optional[str],
    datadir: Optional[str],
    pathfinding_service_address: Optional[str],
    routing_mode: Any,
    **kwargs: Any
) -> RaidenService: ...
```