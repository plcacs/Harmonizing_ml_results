import dataclasses
import json
import random
from collections import defaultdict
from unittest.mock import Mock, PropertyMock
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
from raiden.tests.utils.factories import UNIT_CHAIN_ID, make_token_network_registry_address
from raiden.tests.utils.transfer import create_route_state_for_route
from raiden.transfer import node, views
from raiden.transfer.state import ChainState, NettingChannelState
from raiden.transfer.state_change import Block
from raiden.utils.keys import privatekey_to_address
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import Address, AddressMetadata, BlockIdentifier, BlockNumber, BlockTimeout, ChannelID, Dict, List, Optional, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, Tuple
from raiden_contracts.utils.type_aliases import ChainID

RoutesDict = Dict[TokenAddress, Dict[Tuple[Address, Address], List[List[RaidenService]]]]

class MockJSONRPCClient:
    balances_mapping: Dict[Address, TokenAmount]
    chain_id: ChainID
    address: Address

    def __init__(self, address: Address) -> None: ...
    @staticmethod
    def can_query_state_for_block(block_identifier: BlockIdentifier) -> bool: ...
    def gas_price(self) -> int: ...
    def balance(self, address: Address) -> TokenAmount: ...

class MockTokenNetworkProxy:
    client: MockJSONRPCClient

    def __init__(self, client: MockJSONRPCClient) -> None: ...
    @staticmethod
    def detail_participants(participant1: Address, participant2: Address, block_identifier: BlockIdentifier, channel_identifier: ChannelID) -> None: ...

class MockPaymentChannel:
    token_network: MockTokenNetworkProxy

    def __init__(self, token_network: MockTokenNetworkProxy, channel_id: ChannelID) -> None: ...

class MockProxyManager:
    client: MockJSONRPCClient
    token_network: MockTokenNetworkProxy
    mocked_addresses: Dict[str, Address]

    def __init__(self, node_address: Address, mocked_addresses: Optional[Dict[str, Address]] = None) -> None: ...
    def payment_channel(self, channel_state: any, block_identifier: BlockIdentifier) -> MockPaymentChannel: ...
    def token_network_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...
    def secret_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...
    def user_deposit(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...
    def service_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...
    def one_to_n(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...
    def monitoring_service(self, address: Address, block_identifier: BlockIdentifier) -> Mock: ...

class MockChannelState:
    settle_transaction: Optional[any]
    close_transaction: Optional[any]
    canonical_identifier: any
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
    message_handler: Optional[any]
    routing_mode: RoutingMode
    config: RaidenConfig
    default_user_deposit: Mock
    default_registry: Mock
    default_one_to_n_address: Address
    default_msc_address: Address
    targets_to_identifiers_to_statuses: defaultdict[any, Dict[any, any]]
    route_to_feedback_token: Dict[any, any]
    notifications: Dict[any, Notification]
    wal: WriteAheadLog
    transport: Mock
    pfs_proxy: PFSProxy

    def __init__(self, message_handler: Optional[any] = None, state_transition: Optional[any] = None, private_key: Optional[bytes] = None) -> None: ...
    def add_notification(self, notification: Notification, click_opts: Optional[Dict[str, any]] = None) -> None: ...
    def on_messages(self, messages: List[any]) -> None: ...
    def handle_and_track_state_changes(self, state_changes: List[any]) -> None: ...
    def handle_state_changes(self, state_changes: List[any]) -> None: ...
    def sign(self, message: any) -> None: ...
    def stop(self) -> None: ...
    def __del__(self) -> None: ...

def make_raiden_service_mock(token_network_registry_address: TokenNetworkRegistryAddress, token_network_address: TokenNetworkAddress, channel_identifier: ChannelID, partner: Address) -> MockRaidenService: ...

def mocked_failed_response(error: Exception, status_code: int = 200) -> Mock: ...

def mocked_json_response(response_data: Optional[Dict[str, any]] = None, status_code: int = 200) -> Mock: ...

class MockEth:
    chain_id: ChainID

    def __init__(self, chain_id: ChainID) -> None: ...
    def get_block(self, block_identifier: BlockIdentifier) -> Dict[str, any]: ...
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
    routes: defaultdict[TokenAddress, defaultdict[Tuple[Address, Address], List[List[RaidenService]]]]

    def __init__(self, pfs_info: PFSInfo) -> None: ...
    def get_pfs_info(self, url: str) -> PFSInfo: ...
    def on_new_block(self, block: Block) -> None: ...
    def update_info(self, confirmed_block_number: Optional[BlockNumber] = None, price: Optional[TokenAmount] = None, matrix_server: Optional[str] = None) -> None: ...
    def query_address_metadata(self, pfs_config: PFSConfig, user_address: Address) -> AddressMetadata: ...
    @staticmethod
    def _get_app_address_metadata(app: any) -> Tuple[Address, AddressMetadata]: ...
    def add_apps(self, apps: List[any], add_pfs_config: bool = True) -> None: ...
    def set_route(self, token_address: TokenAddress, route: List[RaidenService]) -> None: ...
    def reset_routes(self, token_address: Optional[TokenAddress] = None) -> None: ...
    def get_best_routes_pfs(self, chain_state: ChainState, token_network_address: TokenNetworkAddress, one_to_n_address: Address, from_address: Address, to_address: Address, amount: TokenAmount, previous_address: Optional[Address], pfs_config: PFSConfig, privkey: bytes, pfs_wait_for_block: bool) -> Tuple[Optional[str], List[any], Optional[any]]: ...

def make_pfs_config() -> PFSConfig: ...