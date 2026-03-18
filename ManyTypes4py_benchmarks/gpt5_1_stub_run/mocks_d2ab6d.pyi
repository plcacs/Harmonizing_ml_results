from typing import Any, Dict, List, Optional, Tuple

from raiden.api.objects import Notification
from raiden.contracts.utils.type_aliases import ChainID as ChainID  # type: ignore[attr-defined]
from raiden.network.pathfinding import PFSConfig, PFSInfo, PFSProxy
from raiden.raiden_service import RaidenService
from raiden.settings import RaidenConfig
from raiden.transfer.state import ChainState
from raiden.transfer.state_change import Block
from raiden.utils.typing import (
    Address,
    AddressMetadata,
    BlockNumber,
    BlockTimeout,
    ChannelID,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
)

RoutesDict = Dict[TokenAddress, Dict[Tuple[Address, Address], List[List[RaidenService]]]]


class MockJSONRPCClient:
    def __init__(self, address: Any) -> None: ...
    @staticmethod
    def can_query_state_for_block(block_identifier: Any) -> bool: ...
    def gas_price(self) -> int: ...
    def balance(self, address: Any) -> Any: ...


class MockTokenNetworkProxy:
    def __init__(self, client: Any) -> None: ...
    @staticmethod
    def detail_participants(
        participant1: Any,
        participant2: Any,
        block_identifier: Any,
        channel_identifier: Any,
    ) -> None: ...


class MockPaymentChannel:
    def __init__(self, token_network: Any, channel_id: Any) -> None: ...


class MockProxyManager:
    def __init__(self, node_address: Any, mocked_addresses: Optional[Dict[str, Any]] = ...) -> None: ...
    def payment_channel(self, channel_state: Any, block_identifier: Any) -> MockPaymentChannel: ...
    def token_network_registry(self, address: TokenNetworkRegistryAddress, block_identifier: Any) -> Any: ...
    def secret_registry(self, address: Address, block_identifier: Any) -> Any: ...
    def user_deposit(self, address: Address, block_identifier: Any) -> Any: ...
    def service_registry(self, address: Address, block_identifier: Any) -> Any: ...
    def one_to_n(self, address: Address, block_identifier: Any) -> Any: ...
    def monitoring_service(self, address: Address, block_identifier: Any) -> Any: ...


class MockChannelState:
    def __init__(self) -> None: ...


class MockTokenNetwork:
    def __init__(self) -> None: ...


class MockTokenNetworkRegistry:
    def __init__(self) -> None: ...


class MockChainState:
    def __init__(self) -> None: ...


class MockRaidenService:
    def __init__(
        self,
        message_handler: Optional[Any] = ...,
        state_transition: Optional[Any] = ...,
        private_key: Optional[Any] = ...,
    ) -> None: ...
    def add_notification(self, notification: Notification, click_opts: Optional[Dict[str, Any]] = ...) -> None: ...
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


def mocked_failed_response(error: Any, status_code: int = ...) -> Any: ...


def mocked_json_response(response_data: Optional[Dict[str, Any]] = ..., status_code: int = ...) -> Any: ...


class MockEth:
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
    def __init__(self, pfs_info: PFSInfo) -> None: ...
    def get_pfs_info(self, url: Any) -> PFSInfo: ...
    def on_new_block(self, block: Block) -> None: ...
    def update_info(
        self,
        confirmed_block_number: Optional[BlockNumber] = ...,
        price: Optional[TokenAmount] = ...,
        matrix_server: Optional[str] = ...,
    ) -> None: ...
    def query_address_metadata(self, pfs_config: PFSConfig, user_address: Address) -> AddressMetadata: ...
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
        privkey: Any,
        pfs_wait_for_block: Any,
    ) -> Tuple[Optional[str], List[Any], Optional[Any]]: ...


def make_pfs_config() -> PFSConfig: ...