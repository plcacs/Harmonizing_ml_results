import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import ParseResult

import click
import structlog
from eth_typing import URI
from eth_utils import Address as EthAddress
from web3 import Web3

from raiden.accounts import AccountManager
from raiden.api.rest import APIServer, RestAPI
from raiden.constants import (
    BLOCK_ID_LATEST,
    Environment,
    Networks,
    RoutingMode,
)
from raiden.exceptions import ConfigurationError, RaidenError
from raiden.message_handler import MessageHandler
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.network.rpc.client import JSONRPCClient
from raiden.network.transport import MatrixTransport
from raiden.raiden_event_handler import EventHandler, PFSFeedbackEventHandler, RaidenEventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import (
    DEFAULT_HTTP_SERVER_PORT,
    DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS,
    MATRIX_AUTO_SELECT_SERVER,
    MatrixTransportConfig,
    PythonApiConfig,
    RaidenConfig,
    RestApiConfig,
)
from raiden.ui.checks import (
    check_ethereum_chain_id,
    check_ethereum_confirmed_block_is_not_pruned,
    check_ethereum_has_accounts,
    check_sql_version,
    check_synced,
)
from raiden.ui.prompt import (
    prompt_account,
    unlock_account_with_passwordfile,
    unlock_account_with_passwordprompt,
)
from raiden.ui.startup import (
    load_deployed_contracts_data,
    load_deployment_addresses_from_contracts,
    load_deployment_addresses_from_udc,
    raiden_bundle_from_contracts_deployment,
    services_bundle_from_contracts_deployment,
)
from raiden.utils.cli import get_matrix_servers
from raiden.utils.formatting import pex, to_checksum_address
from raiden.utils.http import split_endpoint
from raiden.utils.mediation_fees import prepare_mediation_fee_config
from raiden.utils.typing import (
    Address,
    BlockNumber,
    BlockTimeout,
    ChainID,
    Endpoint,
    FeeAmount,
    PrivateKey,
    ProportionalFeeAmount,
    TokenAddress,
    UserDepositAddress,
)
from raiden_contracts.constants import CONTRACT_TOKEN_NETWORK_REGISTRY, ID_TO_CHAINNAME
from raiden_contracts.contract_manager import ContractDevEnvironment, ContractManager

log: structlog.stdlib.BoundLogger = ...

def fetch_available_matrix_servers(
    transport_config: MatrixTransportConfig,
    environment_type: Environment,
) -> None: ...

def get_account_and_private_key(
    account_manager: AccountManager,
    address: Optional[str],
    password_file: Optional[str],
) -> Tuple[Address, PrivateKey]: ...

def get_smart_contracts_start_at(chain_id: ChainID) -> BlockNumber: ...

def get_min_reveal_timeout(chain_id: ChainID) -> BlockTimeout: ...

def rpc_normalized_endpoint(eth_rpc_endpoint: str) -> URI: ...

def start_api_server(
    rpc_client: JSONRPCClient,
    config: RestApiConfig,
    eth_rpc_endpoint: str,
) -> APIServer: ...

def setup_raiden_config(
    eth_rpc_endpoint: str,
    api_address: str,
    rpc: bool,
    rpccorsdomain: Optional[str],
    console: bool,
    web_ui: bool,
    matrix_server: str,
    chain_id: ChainID,
    environment_type: Environment,
    development_environment: ContractDevEnvironment,
    unrecoverable_error_should_crash: bool,
    pathfinding_max_paths: int,
    enable_monitoring: bool,
    resolver_endpoint: Optional[str],
    default_reveal_timeout: BlockTimeout,
    default_settle_timeout: BlockTimeout,
    flat_fee: Optional[Dict[TokenAddress, FeeAmount]],
    proportional_fee: Optional[Dict[TokenAddress, ProportionalFeeAmount]],
    proportional_imbalance_fee: Optional[Dict[TokenAddress, ProportionalFeeAmount]],
    blockchain_query_interval: float,
    cap_mediation_fees: bool,
    enable_tracing: bool,
    **kwargs: Any,
) -> RaidenConfig: ...

def run_raiden_service(
    config: RaidenConfig,
    eth_rpc_endpoint: str,
    address: Optional[str],
    keystore_path: str,
    gas_price: Optional[str],
    user_deposit_contract_address: Optional[UserDepositAddress],
    sync_check: bool,
    password_file: Optional[str],
    datadir: Optional[str],
    pathfinding_service_address: Optional[Address],
    routing_mode: RoutingMode,
    **kwargs: Any,
) -> RaidenService: ...