```python
import os
from pathlib import Path
from typing import Any, Callable, Optional, TextIO, Tuple
from urllib.parse import ParseResult
import click
import structlog
from eth_typing import URI
from web3 import Web3
from raiden.accounts import AccountManager
from raiden.api.rest import APIServer, RestAPI
from raiden.constants import (
    BLOCK_ID_LATEST,
    CHAIN_TO_MIN_REVEAL_TIMEOUT,
    DOC_URL,
    GENESIS_BLOCK_NUMBER,
    RAIDEN_DB_VERSION,
    Environment,
    EthereumForks,
    GoerliForks,
    KovanForks,
    Networks,
    RinkebyForks,
    RopstenForks,
    RoutingMode,
)
from raiden.exceptions import ConfigurationError, RaidenError
from raiden.message_handler import MessageHandler
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.rpc.client import JSONRPCClient
from raiden.network.transport import MatrixTransport
from raiden.raiden_event_handler import EventHandler, PFSFeedbackEventHandler, RaidenEventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import (
    DEFAULT_HTTP_SERVER_PORT,
    DEFAULT_MATRIX_KNOWN_SERVERS,
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

log: Any = ...

def fetch_available_matrix_servers(
    transport_config: Any,
    environment_type: Any,
) -> None: ...

def get_account_and_private_key(
    account_manager: AccountManager,
    address: Any,
    password_file: Any,
) -> Tuple[Any, Any]: ...

def get_smart_contracts_start_at(chain_id: Any) -> Any: ...

def get_min_reveal_timeout(chain_id: Any) -> Any: ...

def rpc_normalized_endpoint(eth_rpc_endpoint: Any) -> URI: ...

def start_api_server(
    rpc_client: Any,
    config: Any,
    eth_rpc_endpoint: Any,
) -> APIServer: ...

def setup_raiden_config(
    eth_rpc_endpoint: Any,
    api_address: Any,
    rpc: Any,
    rpccorsdomain: Any,
    console: Any,
    web_ui: Any,
    matrix_server: Any,
    chain_id: Any,
    environment_type: Any,
    development_environment: Any,
    unrecoverable_error_should_crash: Any,
    pathfinding_max_paths: Any,
    enable_monitoring: Any,
    resolver_endpoint: Any,
    default_reveal_timeout: Any,
    default_settle_timeout: Any,
    flat_fee: Any,
    proportional_fee: Any,
    proportional_imbalance_fee: Any,
    blockchain_query_interval: Any,
    cap_mediation_fees: Any,
    enable_tracing: Any,
    **kwargs: Any,
) -> RaidenConfig: ...

def run_raiden_service(
    config: RaidenConfig,
    eth_rpc_endpoint: Any,
    address: Any,
    keystore_path: Any,
    gas_price: Any,
    user_deposit_contract_address: Any,
    sync_check: Any,
    password_file: Any,
    datadir: Any,
    pathfinding_service_address: Any,
    routing_mode: Any,
    **kwargs: Any,
) -> RaidenService: ...
```