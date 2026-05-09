import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from urllib.parse import ParseResult
import click
import structlog
from eth_typing import URI
from eth_utils import ChainID
from web3 import Web3
from raiden.accounts import AccountManager
from raiden.api.rest import APIServer, RestAPI
from raiden.constants import BlockNumber, BlockTimeout, Environment, EthereumForks, GoerliForks, KovanForks, Networks, RinkebyForks, RopstenForks, RoutingMode
from raiden.exceptions import ConfigurationError, RaidenError
from raiden.message_handler import MessageHandler
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.rpc.client import JSONRPCClient
from raiden.network.transport import MatrixTransport
from raiden.raiden_event_handler import EventHandler, PFSFeedbackEventHandler, RaidenEventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import PythonApiConfig, RaidenConfig, RestApiConfig
from raiden.utils.mediation_fees import FeeConfig
from raiden.utils.typing import Address, BlockNumber, BlockTimeout, ChainID, Endpoint, FeeAmount, Optional, PrivateKey, ProportionalFeeAmount, TokenAddress, Tuple, UserDepositAddress
from raiden_contracts.contract_manager import ContractManager

log = structlog.get_logger(__name__)

def fetch_available_matrix_servers(transport_config: Any, environment_type: Environment) -> None:
    ...

def get_account_and_private_key(account_manager: AccountManager, address: Optional[Address], password_file: Optional[Path]) -> Tuple[Address, bytes]:
    ...

def get_smart_contracts_start_at(chain_id: ChainID) -> BlockNumber:
    ...

def get_min_reveal_timeout(chain_id: ChainID) -> BlockTimeout:
    ...

def rpc_normalized_endpoint(eth_rpc_endpoint: str) -> URI:
    ...

def start_api_server(rpc_client: JSONRPCClient, config: RestApiConfig, eth_rpc_endpoint: str) -> APIServer:
    ...

def setup_raiden_config(eth_rpc_endpoint: str, api_address: Endpoint, rpc: bool, rpccorsdomain: Optional[str], console: bool, web_ui: bool, matrix_server: str, chain_id: ChainID, environment_type: Environment, development_environment: ContractDevEnvironment, unrecoverable_error_should_crash: bool, pathfinding_max_paths: int, enable_monitoring: bool, resolver_endpoint: Optional[str], default_reveal_timeout: BlockTimeout, default_settle_timeout: BlockTimeout, flat_fee: Optional[FeeAmount], proportional_fee: Optional[ProportionalFeeAmount], proportional_imbalance_fee: Optional[ProportionalFeeAmount], blockchain_query_interval: float, cap_mediation_fees: bool, enable_tracing: bool, **kwargs: Any) -> RaidenConfig:
    ...

def run_raiden_service(config: RaidenConfig, eth_rpc_endpoint: str, address: Optional[Address], keystore_path: Path, gas_price: Callable[[Web3, bytes], int], user_deposit_contract_address: Optional[UserDepositAddress], sync_check: bool, password_file: Optional[Path], datadir: Optional[str], pathfinding_service_address: Optional[Address], routing_mode: RoutingMode, **kwargs: Any) -> RaidenService:
    ...