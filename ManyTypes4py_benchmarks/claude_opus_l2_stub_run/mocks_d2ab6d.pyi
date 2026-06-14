import dataclasses
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import click

from raiden.api.objects import Notification
from raiden.constants import Environment, RoutingMode
from raiden.network.pathfinding import PFSConfig, PFSInfo, PFSProxy
from raiden.raiden_service import RaidenService
from raiden.settings import RaidenConfig
from raiden.storage.wal import WriteAheadLog
from raiden.transfer.state import ChainState, NettingChannelState
from raiden.transfer.state_change import Block
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
    balances_mapping: Dict[Address, Any]
    chain_id: ChainID
    address: Address

    def __init__(self, address: Address) -> None: ...

    @staticmethod
    def can_query_state_for_block(block_identifier: BlockIdentifier) -> bool: ...

    def gas_price(self) -> int: ...

    def balance(self, address: Address) -> Any: ...

class MockTokenNetworkProxy:
    client: MockJSONRPCClient

    def __init__(self, client: MockJSONRPCClient) -> None: ...

    @staticmethod
    def detail_participants(
        participant1: Any,
        participant2: Any,
        block_identifier: BlockIdentifier,
        channel_identifier: ChannelID,
    ) -> None: ...

class MockPaymentChannel:
    token_network: MockTokenNetworkProxy

    def __init__(self, token_network: MockTokenNetworkProxy, channel_id: ChannelID) -> None: ...

class MockProxyManager:
    client: MockJSONRPCClient
    token_network: MockTokenNetworkProxy
    mocked_addresses: Dict[str, Address]

    def __init__(
        self,
        node_address: Address,
        mocked_addresses: Optional[Dict[str, Address]] = ...,
    ) -> None: ...

    def payment_channel(
        self, channel_state: NettingChannelState, block_identifier: BlockIdentifier
    ) -> MockPaymentChannel: ...

    def token_network_registry(
        self, address: TokenNetworkRegistryAddress, block_identifier: BlockIdentifier
    ) -> Mock: ...

    def secret_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...

    def user_deposit(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...

    def service_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...

    def one_to_n(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...

    def monitoring_service(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...

class MockChannelState:
    settle_transaction: None
    close_transaction: None
    canonical_identifier: Any
    our_state: Mock
    partner_state: Mock

    def __init__(self) -> None: ...

class MockTokenNetwork:
    channelidentifiers_to_channels: Dict[ChannelID, MockChannelState]
    partneraddresses_to_channelidentifiers: Dict[Address, List[ChannelID]]

    def __init__(self) -> None: ...

class MockTokenNetworkRegistry:
    tokennetworkaddresses_to_tokennetworks: Dict[TokenNetworkAddress, MockTokenNetwork]

    def __init__(self) -> None: ...

class MockChainState:
    block_hash: bytes
    identifiers_to_tokennetworkregistries: Dict[TokenNetworkRegistryAddress, MockTokenNetworkRegistry]

    def __init__(self) -> None: ...

class MockRaidenService:
    privkey: bytes
    address: Address
    rpc_client: MockJSONRPCClient
    proxy_manager: MockProxyManager
    signer: LocalSigner
    message_handler: Any
    routing_mode: RoutingMode
    config: RaidenConfig
    default_user_deposit: Mock
    default_registry: Mock
    default_one_to_n_address: Address
    default_msc_address: Address
    targets_to_identifiers_to_statuses: defaultdict[Any, Dict[Any, Any]]
    route_to_feedback_token: Dict[Any, Any]
    notifications: Dict[Any, Notification]
    wal: Any
    transport: Mock
    pfs_proxy: PFSProxy

    def __init__(
        self,
        message_handler: Any = ...,
        state_transition: Any = ...,
        private_key: Optional[bytes] = ...,
    ) -> None: ...

    def add_notification(
        self, notification: Notification, click_opts: Optional[Dict[str, Any]] = ...
    ) -> None: ...

    def on_messages(self, messages: Any) -> None: ...

    def handle_and_track_state_changes(self, state_changes: Any) -> None: ...

    def handle_state_changes(self, state_changes: Any) -> None: ...

    def sign(self, message: Any) -> None: ...

    def stop(self) -> None: ...

    def __del__(self) -> None: ...

def make_raiden_service_mock(
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_network_address: TokenNetworkAddress,
    channel_identifier: ChannelID,
    partner: Address,
) -> MockRaidenService: ...

def mocked_failed_response(error: Any, status_code: int = ...) -> Mock: ...

def mocked_json_response(response_data: Optional[Dict[str, Any]] = ..., status_code: int = ...) -> Mock: ...

class MockEth:
    chain_id: ChainID

    def __init__(self, chain_id: ChainID) -> None: ...

    def get_block(self, block_identifier: BlockIdentifier) -> Dict[str, Any]: ...

    @property
    def chainId(self) -> ChainID: ...

class MockWeb3:
    eth: MockEth

    def __init__(self, chain_id: ChainID) -> None: ...

class PFSMock:
    PFSCONFIG_MAXIMUM_FEE: TokenAmount
    PFSCONFIG_IOU_TIMEOUT: BlockTimeout
    PFSCONFIG_MAX_PATHS: int

    pfs_info: PFSInfo
    address_to_address_metadata: Dict[Address, AddressMetadata]
    routes: defaultdict[TokenAddress, defaultdict[Tuple[Address, Address], List[Any]]]

    def __init__(self, pfs_info: PFSInfo) -> None: ...

    def get_pfs_info(self, url: str) -> PFSInfo: ...

    def on_new_block(self, block: Block) -> None: ...

    def update_info(
        self,
        confirmed_block_number: Optional[BlockNumber] = ...,
        price: Optional[TokenAmount] = ...,
        matrix_server: Optional[str] = ...,
    ) -> None: ...

    def query_address_metadata(
        self, pfs_config: PFSConfig, user_address: Address
    ) -> AddressMetadata: ...

    @staticmethod
    def _get_app_address_metadata(app: Any) -> Tuple[Address, AddressMetadata]: ...

    def add_apps(self, apps: List[Any], add_pfs_config: bool = ...) -> None: ...

    def set_route(self, token_address: TokenAddress, route: List[Any]) -> None: ...

    def reset_routes(self, token_address: Optional[TokenAddress] = ...) -> None: ...

    def get_best_routes_pfs(
        self,
        chain_state: ChainState,
        token_network_address: TokenNetworkAddress,
        one_to_n_address: Address,
        from_address: Address,
        to_address: Address,
        amount: TokenAmount,
        previous_address: Optional[Address],
        pfs_config: PFSConfig,
        privkey: bytes,
        pfs_wait_for_block: BlockNumber,
    ) -> Tuple[Optional[str], List[Any], Optional[Any]]: ...

def make_pfs_config() -> PFSConfig: ...