```python
import dataclasses
import json
from collections import defaultdict
from unittest.mock import Mock, PropertyMock
from typing import Any, Dict, List, Optional, Tuple, Union

import click
from raiden.api.objects import Notification
from raiden.constants import Environment, RoutingMode
from raiden.exceptions import PFSReturnedError
from raiden.network.pathfinding import PFSConfig, PFSInfo, PFSProxy
from raiden.raiden_service import RaidenService
from raiden.settings import RaidenConfig
from raiden.storage.serialization import JSONSerializer
from raiden.storage.sqlite import SerializedSQLiteStorage
from raiden.storage.wal import WriteAheadLog
from raiden.tests.utils import factories
from raiden.tests.utils.transfer import create_route_state_for_route
from raiden.transfer import node, views
from raiden.transfer.state import ChainState, NettingChannelState
from raiden.transfer.state_change import Block
from raiden.utils.keys import privatekey_to_address
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import (
    Address,
    AddressMetadata,
    BlockIdentifier,
    BlockNumber,
    BlockTimeout,
    ChannelID,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
)
from raiden_contracts.utils.type_aliases import ChainID

RoutesDict = Dict[TokenAddress, Dict[Tuple[Address, Address], List[List[RaidenService]]]]

class MockJSONRPCClient:
    balances_mapping: Dict[Any, Any]
    chain_id: ChainID
    address: Any

    def __init__(self, address: Any) -> None: ...
    @staticmethod
    def can_query_state_for_block(block_identifier: Any) -> bool: ...
    def gas_price(self) -> int: ...
    def balance(self, address: Any) -> Any: ...

class MockTokenNetworkProxy:
    client: Any

    def __init__(self, client: Any) -> None: ...
    @staticmethod
    def detail_participants(
        participant1: Any,
        participant2: Any,
        block_identifier: Any,
        channel_identifier: Any,
    ) -> None: ...

class MockPaymentChannel:
    token_network: Any

    def __init__(self, token_network: Any, channel_id: Any) -> None: ...

class MockProxyManager:
    client: MockJSONRPCClient
    token_network: MockTokenNetworkProxy
    mocked_addresses: Dict[str, Any]

    def __init__(self, node_address: Any, mocked_addresses: Optional[Dict[str, Any]] = ...) -> None: ...
    def payment_channel(self, channel_state: Any, block_identifier: Any) -> MockPaymentChannel: ...
    def token_network_registry(self, address: Any, block_identifier: Any) -> Mock: ...
    def secret_registry(self, address: Any, block_identifier: Any) -> Mock: ...
    def user_deposit(self, address: Any, block_identifier: Any) -> Mock: ...
    def service_registry(self, address: Any, block_identifier: Any) -> Mock: ...
    def one_to_n(self, address: Any, block_identifier: Any) -> Mock: ...
    def monitoring_service(self, address: Any, block_identifier: Any) -> Mock: ...

class MockChannelState:
    settle_transaction: Any
    close_transaction: Any
    canonical_identifier: Any
    our_state: Mock
    partner_state: Mock

    def __init__(self) -> None: ...

class MockTokenNetwork:
    channelidentifiers_to_channels: Dict[Any, Any]
    partneraddresses_to_channelidentifiers: Dict[Any, List[Any]]

    def __init__(self) -> None: ...

class MockTokenNetworkRegistry:
    tokennetworkaddresses_to_tokennetworks: Dict[Any, Any]

    def __init__(self) -> None: ...

class MockChainState:
    block_hash: Any
    identifiers_to_tokennetworkregistries: Dict[Any, Any]

    def __init__(self) -> None: ...

class MockRaidenService:
    privkey: Any
    address: Any
    rpc_client: MockJSONRPCClient
    proxy_manager: MockProxyManager
    signer: LocalSigner
    message_handler: Any
    routing_mode: RoutingMode
    config: RaidenConfig
    default_user_deposit: Mock
    default_registry: Mock
    default_one_to_n_address: Any
    default_msc_address: Any
    targets_to_identifiers_to_statuses: Dict[Any, Dict[Any, Any]]
    route_to_feedback_token: Dict[Any, Any]
    notifications: Dict[Any, Notification]
    transport: Mock
    pfs_proxy: PFSProxy
    wal: WriteAheadLog

    def __init__(
        self,
        message_handler: Any = ...,
        state_transition: Any = ...,
        private_key: Any = ...,
    ) -> None: ...
    def add_notification(self, notification: Notification, click_opts: Optional[Dict[str, Any]] = ...) -> None: ...
    def on_messages(self, messages: Any) -> None: ...
    def handle_and_track_state_changes(self, state_changes: Any) -> None: ...
    def handle_state_changes(self, state_changes: Any) -> None: ...
    def sign(self, message: Any) -> None: ...
    def stop(self) -> None: ...
    def __del__(self) -> None: ...

def make_raiden_service_mock(
    token_network_registry_address: Any,
    token_network_address: Any,
    channel_identifier: Any,
    partner: Any,
) -> MockRaidenService: ...

def mocked_failed_response(error: Any, status_code: int = ...) -> Mock: ...

def mocked_json_response(response_data: Optional[Any] = ..., status_code: int = ...) -> Mock: ...

class MockEth:
    chain_id: Any

    def __init__(self, chain_id: Any) -> None: ...
    def get_block(self, block_identifier: Any) -> Dict[str, Any]: ...
    @property
    def chainId(self) -> Any: ...

class MockWeb3:
    eth: MockEth

    def __init__(self, chain_id: Any) -> None: ...

class PFSMock:
    PFSCONFIG_MAXIMUM_FEE: TokenAmount
    PFSCONFIG_IOU_TIMEOUT: BlockTimeout
    PFSCONFIG_MAX_PATHS: int
    pfs_info: PFSInfo
    address_to_address_metadata: Dict[Any, AddressMetadata]
    routes: Dict[Any, Dict[Tuple[Any, Any], List[Any]]]

    def __init__(self, pfs_info: PFSInfo) -> None: ...
    def get_pfs_info(self, url: Any) -> PFSInfo: ...
    def on_new_block(self, block: Block) -> None: ...
    def update_info(
        self,
        confirmed_block_number: Optional[BlockNumber] = ...,
        price: Optional[TokenAmount] = ...,
        matrix_server: Optional[str] = ...,
    ) -> None: ...
    def query_address_metadata(self, pfs_config: PFSConfig, user_address: Any) -> AddressMetadata: ...
    @staticmethod
    def _get_app_address_metadata(app: Any) -> Tuple[Any, AddressMetadata]: ...
    def add_apps(self, apps: Any, add_pfs_config: bool = ...) -> None: ...
    def set_route(self, token_address: Any, route: Any) -> None: ...
    def reset_routes(self, token_address: Optional[Any] = ...) -> None: ...
    def get_best_routes_pfs(
        self,
        chain_state: ChainState,
        token_network_address: TokenNetworkAddress,
        one_to_n_address: Any,
        from_address: Address,
        to_address: Address,
        amount: TokenAmount,
        previous_address: Optional[Address],
        pfs_config: PFSConfig,
        privkey: Any,
        pfs_wait_for_block: Any,
    ) -> Tuple[Optional[str], List[Any], None]: ...

def make_pfs_config() -> PFSConfig: ...
```