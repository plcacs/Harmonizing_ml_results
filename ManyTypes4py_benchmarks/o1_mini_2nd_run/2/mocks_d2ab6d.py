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
    balances_mapping: Dict[Address, TokenAmount]
    chain_id: ChainID
    address: Address

    def __init__(self, address: Address) -> None:
        self.balances_mapping = {}
        self.chain_id = ChainID(UNIT_CHAIN_ID)
        self.address = address

    @staticmethod
    def can_query_state_for_block(block_identifier: BlockIdentifier) -> bool:
        return True

    def gas_price(self) -> int:
        return 1000000000

    def balance(self, address: Address) -> TokenAmount:
        return self.balances_mapping[address]


class MockTokenNetworkProxy:
    client: MockJSONRPCClient

    def __init__(self, client: MockJSONRPCClient) -> None:
        self.client = client

    @staticmethod
    def detail_participants(
        participant1: Address, participant2: Address, block_identifier: BlockIdentifier, channel_identifier: ChannelID
    ) -> Optional[None]:
        return None


class MockPaymentChannel:
    token_network: MockTokenNetworkProxy
    channel_id: ChannelID

    def __init__(self, token_network: MockTokenNetworkProxy, channel_id: ChannelID) -> None:
        self.token_network = token_network
        self.channel_id = channel_id


class MockProxyManager:
    client: MockJSONRPCClient
    token_network: MockTokenNetworkProxy
    mocked_addresses: Dict[str, Address]

    def __init__(self, node_address: Address, mocked_addresses: Optional[Dict[str, Address]] = None) -> None:
        self.client = MockJSONRPCClient(node_address)
        self.token_network = MockTokenNetworkProxy(client=self.client)
        self.mocked_addresses = mocked_addresses or {}

    def payment_channel(
        self, channel_state: 'ChannelState', block_identifier: BlockIdentifier
    ) -> MockPaymentChannel:
        return MockPaymentChannel(
            self.token_network, channel_state.canonical_identifier.channel_identifier
        )

    def token_network_registry(
        self, address: Address, block_identifier: BlockIdentifier
    ) -> Mock:
        registry = Mock(address=address)
        registry.get_secret_registry_address.return_value = self.mocked_addresses.get(
            'SecretRegistry', factories.make_address()
        )
        return registry

    def secret_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock:
        return Mock(address=address)

    def user_deposit(self, address: Address, block_identifier: BlockIdentifier) -> Mock:
        user_deposit = Mock()
        user_deposit.monitoring_service_address.return_value = self.mocked_addresses.get(
            'MonitoringService', bytes(20)
        )
        user_deposit.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        user_deposit.one_to_n_address.return_value = self.mocked_addresses.get('OneToN', bytes(20))
        user_deposit.service_registry_address.return_value = self.mocked_addresses.get(
            'ServiceRegistry', bytes(20)
        )
        return user_deposit

    def service_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock:
        service_registry = Mock()
        service_registry.address = self.mocked_addresses.get('ServiceRegistry', bytes(20))
        service_registry.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        return service_registry

    def one_to_n(self, address: Address, block_identifier: BlockIdentifier) -> Mock:
        one_to_n = Mock()
        one_to_n.address = self.mocked_addresses.get('MonitoringService', bytes(20))
        one_to_n.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        return one_to_n

    def monitoring_service(
        self, address: Address, block_identifier: BlockIdentifier
    ) -> Mock:
        monitoring_service = Mock()
        monitoring_service.address = self.mocked_addresses.get('MonitoringService', bytes(20))
        monitoring_service.token_network_registry_address.return_value = self.mocked_addresses.get(
            'TokenNetworkRegistry', bytes(20)
        )
        monitoring_service.service_registry_address.return_value = self.mocked_addresses.get(
            'ServiceRegistry', bytes(20)
        )
        monitoring_service.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        return monitoring_service


class MockChannelState:
    settle_transaction: Optional[Any]
    close_transaction: Optional[Any]
    canonical_identifier: 'CanonicalIdentifier'
    our_state: Mock
    partner_state: Mock

    def __init__(self) -> None:
        self.settle_transaction = None
        self.close_transaction = None
        self.canonical_identifier = factories.make_canonical_identifier()
        self.our_state = Mock()
        self.partner_state = Mock()


class MockTokenNetwork:
    channelidentifiers_to_channels: Dict[ChannelID, MockChannelState]
    partneraddresses_to_channelidentifiers: Dict[Address, List[ChannelID]]

    def __init__(self) -> None:
        self.channelidentifiers_to_channels = {}
        self.partneraddresses_to_channelidentifiers = {}


class MockTokenNetworkRegistry:
    tokennetworkaddresses_to_tokennetworks: Dict[TokenNetworkAddress, MockTokenNetwork]

    def __init__(self) -> None:
        self.tokennetworkaddresses_to_tokennetworks = {}


class MockChainState:
    block_hash: str
    identifiers_to_tokennetworkregistries: Dict[TokenNetworkRegistryAddress, MockTokenNetworkRegistry]

    def __init__(self) -> None:
        self.block_hash = factories.make_block_hash()
        self.identifiers_to_tokennetworkregistries = {}


class MockRaidenService:
    privkey: bytes
    address: Address
    rpc_client: MockJSONRPCClient
    proxy_manager: MockProxyManager
    signer: LocalSigner
    message_handler: Optional[Any]
    routing_mode: RoutingMode
    config: RaidenConfig
    default_user_deposit: Mock
    default_registry: Mock
    default_one_to_n_address: Address
    default_msc_address: Address
    targets_to_identifiers_to_statuses: defaultdict
    route_to_feedback_token: Dict[Any, Any]
    notifications: Dict[Any, Notification]
    wal: WriteAheadLog
    transport: Mock
    pfs_proxy: PFSProxy

    def __init__(
        self,
        message_handler: Optional[Any] = None,
        state_transition: Optional[Any] = None,
        private_key: Optional[bytes] = None,
    ) -> None:
        if private_key is None:
            self.privkey, self.address = factories.make_privkey_address()
        else:
            self.privkey = private_key
            self.address = privatekey_to_address(private_key)
        self.rpc_client = MockJSONRPCClient(self.address)
        self.proxy_manager = MockProxyManager(node_address=self.address)
        self.signer = LocalSigner(self.privkey)
        self.message_handler = message_handler
        self.routing_mode = RoutingMode.PRIVATE
        self.config = RaidenConfig(
            chain_id=self.rpc_client.chain_id,
            environment_type=Environment.DEVELOPMENT,
            pfs_config=make_pfs_config(),
        )
        self.default_user_deposit = Mock()
        self.default_registry = Mock()
        self.default_registry.address = factories.make_address()
        self.default_one_to_n_address = factories.make_address()
        self.default_msc_address = factories.make_address()
        self.targets_to_identifiers_to_statuses = defaultdict(dict)
        self.route_to_feedback_token = {}
        self.notifications = {}
        if state_transition is None:
            state_transition = node.state_transition
        serializer = JSONSerializer()
        initial_state = ChainState(
            pseudo_random_generator=random.Random(),
            block_number=BlockNumber(0),
            block_hash=factories.make_block_hash(),
            our_address=self.rpc_client.address,
            chain_id=self.rpc_client.chain_id,
        )
        wal = WriteAheadLog(
            state=initial_state,
            storage=SerializedSQLiteStorage(':memory:', serializer),
            state_transition=state_transition,
        )
        self.wal = wal
        self.transport = Mock()
        self.pfs_proxy = PFSProxy(make_pfs_config())

    def add_notification(
        self, notification: Notification, click_opts: Optional[Dict[str, Any]] = None
    ) -> None:
        click_opts = click_opts or {}
        click.secho(notification.body, **click_opts)
        self.notifications[notification.id] = notification

    def on_messages(self, messages: List[Any]) -> None:
        if self.message_handler:
            self.message_handler.on_messages(self, messages)

    def handle_and_track_state_changes(self, state_changes: List[Any]) -> None:
        pass

    def handle_state_changes(self, state_changes: List[Any]) -> None:
        pass

    def sign(self, message: Any) -> None:
        message.sign(self.signer)

    def stop(self) -> None:
        self.wal.storage.close()

    def __del__(self) -> None:
        self.stop()


def make_raiden_service_mock(
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_network_address: TokenNetworkAddress,
    channel_identifier: ChannelID,
    partner: Address,
) -> MockRaidenService:
    raiden_service = MockRaidenService()
    chain_state = MockChainState()
    wal = Mock()
    wal.get_current_state.return_value = chain_state
    raiden_service.wal = wal
    token_network = MockTokenNetwork()
    token_network.channelidentifiers_to_channels[channel_identifier] = MockChannelState()
    token_network.partneraddresses_to_channelidentifiers[partner] = [channel_identifier]
    token_network_registry = MockTokenNetworkRegistry()
    tokennetworkaddresses_to_tokennetworks = token_network_registry.tokennetworkaddresses_to_tokennetworks
    tokennetworkaddresses_to_tokennetworks[token_network_address] = token_network
    chain_state.identifiers_to_tokennetworkregistries = {token_network_registry_address: token_network_registry}
    return raiden_service


def mocked_failed_response(error: Exception, status_code: int = 200) -> Mock:
    m = Mock(json=Mock(side_effect=error), status_code=status_code)
    type(m).content = PropertyMock(side_effect=error)
    return m


def mocked_json_response(response_data: Optional[Dict] = None, status_code: int = 200) -> Mock:
    data = response_data or {}
    return Mock(json=Mock(return_value=data), content=json.dumps(data), status_code=status_code)


class MockEth:
    chain_id: ChainID

    def __init__(self, chain_id: ChainID) -> None:
        self.chain_id = chain_id

    def get_block(self, block_identifier: BlockIdentifier) -> Dict[str, Any]:
        return {
            'number': 42,
            'hash': '0x8cb5f5fb0d888c03ec4d13f69d4eb8d604678508a1fa7c1a8f0437d0065b9b67',
        }

    @property
    def chainId(self) -> ChainID:
        return self.chain_id


class MockWeb3:
    eth: MockEth

    def __init__(self, chain_id: ChainID) -> None:
        self.eth = MockEth(chain_id=chain_id)


class PFSMock:
    PFSCONFIG_MAXIMUM_FEE: TokenAmount = TokenAmount(100)
    PFSCONFIG_IOU_TIMEOUT: BlockTimeout = BlockTimeout(5)
    PFSCONFIG_MAX_PATHS: int = 5
    pfs_info: PFSInfo
    address_to_address_metadata: Dict[Address, AddressMetadata]
    routes: Dict[TokenAddress, Dict[Tuple[Address, Address], List[List[RaidenService]]]]

    def __init__(self, pfs_info: PFSInfo) -> None:
        self.pfs_info = pfs_info
        self.address_to_address_metadata = {}
        self.routes = defaultdict(lambda: defaultdict(list))

    def get_pfs_info(self, url: str) -> PFSInfo:
        return self.pfs_info

    def on_new_block(self, block: Block) -> None:
        self.update_info(confirmed_block_number=block.block_number)

    def update_info(
        self,
        confirmed_block_number: Optional[BlockNumber] = None,
        price: Optional[TokenAmount] = None,
        matrix_server: Optional[str] = None,
    ) -> None:
        pfs_info_dict = dataclasses.asdict(self.pfs_info)
        update_dict: Dict[str, Any] = {
            'confirmed_block_number': confirmed_block_number,
            'price': price,
            'matrix_server': matrix_server,
        }
        update_dict = {k: v for k, v in update_dict.items() if v is not None}
        pfs_info_dict.update(update_dict)
        self.pfs_info = PFSInfo(**pfs_info_dict)

    def query_address_metadata(self, pfs_config: PFSConfig, user_address: Address) -> AddressMetadata:
        metadata = self.address_to_address_metadata.get(user_address)
        if not metadata:
            raise PFSReturnedError(message='Address not found', error_code=404, error_details={})
        else:
            return metadata

    @staticmethod
    def _get_app_address_metadata(app: MockRaidenService) -> Tuple[Address, AddressMetadata]:
        address = app.address
        return (address, app.transport.address_metadata)

    def add_apps(self, apps: List[MockRaidenService], add_pfs_config: bool = True) -> None:
        for app in apps:
            address, metadata = self._get_app_address_metadata(app)
            if not all((metadata.get('user_id'), address)):
                raise AssertionError("Cant add app to PFSMock")
            self.address_to_address_metadata[address] = metadata
            if add_pfs_config:
                app.config.pfs_config = PFSConfig(
                    info=self.pfs_info,
                    iou_timeout=self.PFSCONFIG_IOU_TIMEOUT,
                    max_paths=self.PFSCONFIG_MAX_PATHS,
                    maximum_fee=self.PFSCONFIG_MAXIMUM_FEE,
                )

    def set_route(self, token_address: TokenAddress, route: List[RaidenService]) -> None:
        if len(route) > 1:
            from_address: Address = route[0].address
            to_address: Address = route[-1].address
            self.routes[token_address][(from_address, to_address)].append(route)

    def reset_routes(self, token_address: Optional[TokenAddress] = None) -> None:
        if token_address is None:
            self.routes.clear()
        else:
            self.routes[token_address].clear()

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
        pfs_wait_for_block: bool,
    ) -> Tuple[Optional[str], List[Any], Optional[Any]]:
        token_network = views.get_token_network_by_address(chain_state, token_network_address)
        if token_network is None:
            return ('No route found', [], None)
        token_address: TokenAddress = token_network.token_address
        routes_apps: List[List[RaidenService]] = self.routes[token_address][(from_address, to_address)]
        if not routes_apps:
            return ('No route found', [], None)
        paths: List[Any] = [create_route_state_for_route(route, token_address) for route in routes_apps]
        return (None, paths, None)


def make_pfs_config() -> PFSConfig:
    return PFSConfig(
        info=PFSInfo(
            url='mock-address',
            chain_id=UNIT_CHAIN_ID,
            token_network_registry_address=make_token_network_registry_address(),
            user_deposit_address=factories.make_address(),
            payment_address=factories.make_address(),
            confirmed_block_number=BlockNumber(100),
            message='',
            operator='',
            version='',
            price=TokenAmount(0),
            matrix_server='http://matrix.example.com',
        ),
        maximum_fee=TokenAmount(100),
        iou_timeout=BlockTimeout(100),
        max_paths=5,
    )
