import dataclasses
import json
import random
from collections import defaultdict
from unittest.mock import Mock, PropertyMock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
from raiden.api.objects import Notification
from raiden.constants import Environment, RoutingMode
from raiden.exceptions import PFSReturnedError
from raiden.network.pathfinding import PFSConfig, PFSInfo, PFSProxy
from raiden.raiden_service import RaidenService
from raiden.settings import RaidenConfig
from raiden.storage.sqlite import SerializedSQLiteStorage
from raiden.tests.utils import factories
from raiden.tests.utils.transfer import create_route_state_for_route
from raiden.transfer.state import ChainState, NettingChannelState
from raiden.utils.typing import (
    Address,
    AddressMetadata,
    BlockIdentifier,
    BlockNumber,
    BlockTimeout,
    ChannelID,
    Dict,
    List,
    Optional,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    Tuple,
)
from raiden_contracts.utils.type_aliases import ChainID

RoutesDict = Dict[TokenAddress, Dict[Tuple[Address, Address], List[List[RaidenService]]]]

class MockJSONRPCClient:
    def __init__(self, address: Address) -> None:
        ...
    
    def can_query_state_for_block(self, block_identifier: BlockIdentifier) -> bool:
        ...
    
    def gas_price(self) -> int:
        ...
    
    def balance(self, address: Address) -> int:
        ...

class MockTokenNetworkProxy:
    def __init__(self, client: MockJSONRPCClient) -> None:
        ...
    
    @staticmethod
    def detail_participants(
        participant1: Address,
        participant2: Address,
        block_identifier: BlockIdentifier,
        channel_identifier: ChannelID,
    ) -> None:
        ...

class MockPaymentChannel:
    def __init__(self, token_network: MockTokenNetworkProxy, channel_id: ChannelID) -> None:
        ...

class MockProxyManager:
    def __init__(
        self,
        node_address: Address,
        mocked_addresses: Optional[Dict[Address, Address]] = None,
    ) -> None:
        ...
    
    def payment_channel(
        self,
        channel_state: NettingChannelState,
        block_identifier: BlockIdentifier,
    ) -> MockPaymentChannel:
        ...
    
    def token_network_registry(
        self,
        address: Address,
        block_identifier: BlockIdentifier,
    ) -> Mock:
        ...
    
    def secret_registry(
        self,
        address: Address,
        block_identifier: BlockIdentifier,
    ) -> Mock:
        ...
    
    def user_deposit(
        self,
        address: Address,
        block_identifier: BlockIdentifier,
    ) -> Mock:
        ...
    
    def service_registry(
        self,
        address: Address,
        block_identifier: BlockIdentifier,
    ) -> Mock:
        ...
    
    def one_to_n(
        self,
        address: Address,
        block_identifier: BlockIdentifier,
    ) -> Mock:
        ...
    
    def monitoring_service(
        self,
        address: Address,
        block_identifier: BlockIdentifier,
    ) -> Mock:
        ...

class MockChannelState:
    def __init__(self) -> None:
        ...

class MockTokenNetwork:
    def __init__(self) -> None:
        ...

class MockTokenNetworkRegistry:
    def __init__(self) -> None:
        ...

class MockChainState:
    def __init__(self) -> None:
        ...

class MockRaidenService:
    def __init__(
        self,
        message_handler: Optional[Any] = None,
        state_transition: Optional[Callable] = None,
        private_key: Optional[bytes] = None,
    ) -> None:
        ...
    
    def add_notification(
        self,
        notification: Notification,
        click_opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...
    
    def on_messages(self, messages: List[Any]) -> None:
        ...
    
    def handle_and_track_state_changes(self, state_changes: List[Any]) -> None:
        ...
    
    def handle_state_changes(self, state_changes: List[Any]) -> None:
        ...
    
    def sign(self, message: Any) -> None:
        ...
    
    def stop(self) -> None:
        ...
    
    def __del__(self) -> None:
        ...

def make_raiden_service_mock(
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_network_address: TokenNetworkAddress,
    channel_identifier: ChannelID,
    partner: Address,
) -> MockRaidenService:
    ...

def mocked_failed_response(
    error: Exception,
    status_code: int = 200,
) -> Mock:
    ...

def mocked_json_response(
    response_data: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
) -> Mock:
    ...

class MockEth:
    def __init__(self, chain_id: ChainID) -> None:
        ...
    
    def get_block(self, block_identifier: BlockIdentifier) -> Dict[str, Any]:
        ...

class MockWeb3:
    def __init__(self, chain_id: ChainID) -> None:
        ...

class PFSMock:
    PFSCONFIG_MAXIMUM_FEE: TokenAmount
    PFSCONFIG_IOU_TIMEOUT: BlockTimeout
    PFSCONFIG_MAX_PATHS: int

    def __init__(self, pfs_info: PFSInfo) -> None:
        ...
    
    def get_pfs_info(self, url: str) -> PFSInfo:
        ...
    
    def on_new_block(self, block: Block) -> None:
        ...
    
    def update_info(
        self,
        confirmed_block_number: Optional[BlockNumber] = None,
        price: Optional[TokenAmount] = None,
        matrix_server: Optional[str] = None,
    ) -> None:
        ...
    
    def query_address_metadata(self, pfs_config: PFSConfig, user_address: Address) -> AddressMetadata:
        ...
    
    @staticmethod
    def _get_app_address_metadata(app: Any) -> Tuple[Address, AddressMetadata]:
        ...
    
    def add_apps(
        self,
        apps: List[Any],
        add_pfs_config: bool = True,
    ) -> None:
        ...
    
    def set_route(self, token_address: TokenAddress, route: List[Any]) -> None:
        ...
    
    def reset_routes(self, token_address: Optional[TokenAddress] = None) -> None:
        ...
    
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
    ) -> Tuple[Optional[str], List[Any], Optional[str]]:
        ...

def make_pfs_config() -> PFSConfig:
    ...