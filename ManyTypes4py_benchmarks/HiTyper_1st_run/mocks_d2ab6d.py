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

    def __init__(self, address) -> None:
        self.balances_mapping = {}
        self.chain_id = ChainID(UNIT_CHAIN_ID)
        self.address = address

    @staticmethod
    def can_query_state_for_block(block_identifier: raiden.utils.MessageID) -> bool:
        return True

    def gas_price(self) -> int:
        return 1000000000

    def balance(self, address: Union[str, bytes, int]):
        return self.balances_mapping[address]

class MockTokenNetworkProxy:

    def __init__(self, client) -> None:
        self.client = client

    @staticmethod
    def detail_participants(participant1: Union[int, bytes], participant2: Union[int, bytes], block_identifier: Union[int, bytes], channel_identifier: Union[int, bytes]) -> None:
        return None

class MockPaymentChannel:

    def __init__(self, token_network, channel_id) -> None:
        self.token_network = token_network

class MockProxyManager:

    def __init__(self, node_address, mocked_addresses=None) -> None:
        self.client = MockJSONRPCClient(node_address)
        self.token_network = MockTokenNetworkProxy(client=self.client)
        self.mocked_addresses = mocked_addresses or {}

    def payment_channel(self, channel_state: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.transfer.state.NettingChannelState], block_identifier: raiden.utils.MessageID) -> MockPaymentChannel:
        return MockPaymentChannel(self.token_network, channel_state.canonical_identifier.channel_identifier)

    def token_network_registry(self, address: Union[raiden.utils.Address, list[str], raiden.utils.TokenAddress], block_identifier: raiden.utils.MessageID) -> Mock:
        registry = Mock(address=address)
        registry.get_secret_registry_address.return_value = self.mocked_addresses.get('SecretRegistry', factories.make_address())
        return registry

    def secret_registry(self, address: Union[raiden.utils.Address, raiden.utils.BlockSpecification, raiden.utils.BlockIdentifier], block_identifier: raiden.utils.MessageID) -> Mock:
        return Mock(address=address)

    def user_deposit(self, address: raiden.utils.MessageID, block_identifier: raiden.utils.MessageID) -> Mock:
        user_deposit = Mock()
        user_deposit.monitoring_service_address.return_value = self.mocked_addresses.get('MonitoringService', bytes(20))
        user_deposit.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        user_deposit.one_to_n_address.return_value = self.mocked_addresses.get('OneToN', bytes(20))
        user_deposit.service_registry_address.return_value = self.mocked_addresses.get('ServiceRegistry', bytes(20))
        return user_deposit

    def service_registry(self, address: Union[raiden.utils.Address, list[str], raiden.utils.BlockSpecification], block_identifier: raiden.utils.MessageID) -> Mock:
        service_registry = Mock()
        service_registry.address = self.mocked_addresses.get('ServiceRegistry', bytes(20))
        service_registry.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        return service_registry

    def one_to_n(self, address: Union[list[str], raiden.utils.Address, raiden.utils.TokenNetworkAddress], block_identifier: raiden.utils.MessageID) -> Mock:
        one_to_n = Mock()
        one_to_n.address = self.mocked_addresses.get('MonitoringService', bytes(20))
        one_to_n.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        return one_to_n

    def monitoring_service(self, address: Union[raiden.utils.Address, list[str], raiden.utils.BlockIdentifier], block_identifier: raiden.utils.MessageID) -> Mock:
        monitoring_service = Mock()
        monitoring_service.address = self.mocked_addresses.get('MonitoringService', bytes(20))
        monitoring_service.token_network_registry_address.return_value = self.mocked_addresses.get('TokenNetworkRegistry', bytes(20))
        monitoring_service.service_registry_address.return_value = self.mocked_addresses.get('ServiceRegistry', bytes(20))
        monitoring_service.token_address.return_value = self.mocked_addresses.get('Token', bytes(20))
        return monitoring_service

class MockChannelState:

    def __init__(self) -> None:
        self.settle_transaction = None
        self.close_transaction = None
        self.canonical_identifier = factories.make_canonical_identifier()
        self.our_state = Mock()
        self.partner_state = Mock()

class MockTokenNetwork:

    def __init__(self) -> None:
        self.channelidentifiers_to_channels = {}
        self.partneraddresses_to_channelidentifiers = {}

class MockTokenNetworkRegistry:

    def __init__(self) -> None:
        self.tokennetworkaddresses_to_tokennetworks = {}

class MockChainState:

    def __init__(self) -> None:
        self.block_hash = factories.make_block_hash()
        self.identifiers_to_tokennetworkregistries = {}

class MockRaidenService:

    def __init__(self, message_handler=None, state_transition=None, private_key=None) -> None:
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
        self.config = RaidenConfig(chain_id=self.rpc_client.chain_id, environment_type=Environment.DEVELOPMENT, pfs_config=make_pfs_config())
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
        initial_state = ChainState(pseudo_random_generator=random.Random(), block_number=BlockNumber(0), block_hash=factories.make_block_hash(), our_address=self.rpc_client.address, chain_id=self.rpc_client.chain_id)
        wal = WriteAheadLog(state=initial_state, storage=SerializedSQLiteStorage(':memory:', serializer), state_transition=state_transition)
        self.wal = wal
        self.transport = Mock()
        self.pfs_proxy = PFSProxy(make_pfs_config())

    def add_notification(self, notification: Union[str, typing.Callable], click_opts: Union[None, str, bool]=None) -> None:
        click_opts = click_opts or {}
        click.secho(notification.body, **click_opts)
        self.notifications[notification.id] = notification

    def on_messages(self, messages: Union[str, raiden.messages.Message]) -> None:
        if self.message_handler:
            self.message_handler.on_messages(self, messages)

    def handle_and_track_state_changes(self, state_changes: Union[raiden.transfer.architecture.State.Change, dict[str, typing.Any]]) -> None:
        pass

    def handle_state_changes(self, state_changes: Union[raiden.transfer.architecture.State.Change, bool]) -> None:
        pass

    def sign(self, message: bytes) -> None:
        message.sign(self.signer)

    def stop(self) -> None:
        self.wal.storage.close()

    def __del__(self) -> None:
        self.stop()

def make_raiden_service_mock(token_network_registry_address: Union[raiden.utils.TokenAddress, raiden.transfer.state.ChainState, raiden.utils.TokenNetworkRegistryAddress], token_network_address: Union[raiden.utils.TokenAddress, raiden.utils.Address, raiden.utils.TokenNetworkAddress], channel_identifier: Union[raiden.utils.TokenNetworkRegistryAddress, raiden.transfer.state.ChainState, raiden.utils.TokenAddress], partner: Union[raiden.utils.BlockTimeout, raiden.utils.TokenAmount, raiden.transfer.state.TokenNetworkState]) -> MockRaidenService:
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

def mocked_failed_response(error: Union[str, bool, None], status_code: int=200) -> Mock:
    m = Mock(json=Mock(side_effect=error), status_code=status_code)
    type(m).content = PropertyMock(side_effect=error)
    return m

def mocked_json_response(response_data: Union[None, int, str, constants.HttpStatusCode]=None, status_code: int=200) -> Mock:
    data = response_data or {}
    return Mock(json=Mock(return_value=data), content=json.dumps(data), status_code=status_code)

class MockEth:

    def __init__(self, chain_id) -> None:
        self.chain_id = chain_id

    def get_block(self, block_identifier: Union[raiden.transfer.state.BalanceProofUnsignedState, str]) -> dict[typing.Text, typing.Union[int,typing.Text]]:
        return {'number': 42, 'hash': '0x8cb5f5fb0d888c03ec4d13f69d4eb8d604678508a1fa7c1a8f0437d0065b9b67'}

    @property
    def chainId(self):
        return self.chain_id

class MockWeb3:

    def __init__(self, chain_id) -> None:
        self.eth = MockEth(chain_id=chain_id)

class PFSMock:
    PFSCONFIG_MAXIMUM_FEE = TokenAmount(100)
    PFSCONFIG_IOU_TIMEOUT = BlockTimeout(5)
    PFSCONFIG_MAX_PATHS = 5

    def __init__(self, pfs_info: Union[dict[str, typing.Any], list[str], dict]) -> None:
        self.pfs_info = pfs_info
        self.address_to_address_metadata = {}
        self.routes = defaultdict(lambda: defaultdict(list))

    def get_pfs_info(self, url: str):
        return self.pfs_info

    def on_new_block(self, block: Union[blockchain.block.Block, dict]) -> None:
        self.update_info(confirmed_block_number=block.block_number)

    def update_info(self, confirmed_block_number: Union[None, int, float, dict]=None, price: Union[None, int, float, dict]=None, matrix_server: Union[None, int, float, dict]=None) -> None:
        pfs_info_dict = dataclasses.asdict(self.pfs_info)
        update_dict = dict(confirmed_block_number=confirmed_block_number, price=price, matrix_server=matrix_server)
        update_dict = {k: v for k, v in update_dict.items() if v is not None}
        pfs_info_dict.update(update_dict)
        self.pfs_info = PFSInfo(**pfs_info_dict)

    def query_address_metadata(self, pfs_config: Union[str, dict[str, typing.Any], None, raiden.messages.Message], user_address: Union[str, tuple[typing.Literal], raiden.messages.Message]) -> Union[dict[str, typing.Any], dict[str, str], dict[typing.Type, typing.Any]]:
        metadata = self.address_to_address_metadata.get(user_address)
        if not metadata:
            raise PFSReturnedError(message='Address not found', error_code=404, error_details={})
        else:
            return metadata

    @staticmethod
    def _get_app_address_metadata(app: Any) -> tuple:
        address = app.address
        return (address, app.transport.address_metadata)

    def add_apps(self, apps: str, add_pfs_config: bool=True) -> None:
        for app in apps:
            address, metadata = self._get_app_address_metadata(app)
            if not all((metadata.get('user_id'), address)):
                raise AssertionError('Cant add app to PFSMock')
            self.address_to_address_metadata[address] = metadata
            if add_pfs_config is True:
                app.config.pfs_config = PFSConfig(info=self.pfs_info, iou_timeout=self.PFSCONFIG_IOU_TIMEOUT, max_paths=self.PFSCONFIG_MAX_PATHS, maximum_fee=self.PFSCONFIG_MAXIMUM_FEE)

    def set_route(self, token_address: Union[raiden.utils.TokenNetworkAddress, raiden.utils.ABI, raiden.utils.Address], route: Union[list[raiden.utils.TokenAddress], list[raiden.utils.Any]]) -> None:
        if len(route) > 1:
            from_address = route[0].address
            to_address = route[-1].address
            self.routes[token_address][from_address, to_address].append(route)

    def reset_routes(self, token_address: Union[None, raiden.utils.TokenAddress, str]=None) -> None:
        if token_address is None:
            self.routes.clear()
        else:
            self.routes[token_address].clear()

    def get_best_routes_pfs(self, chain_state: Union[raiden.utils.TokenNetworkRegistryAddress, raiden.utils.TokenAddress, raiden.transfer.state.ChainState], token_network_address: Union[raiden.utils.TokenNetworkRegistryAddress, raiden.utils.TokenAddress, raiden.transfer.state.ChainState], one_to_n_address: bool, from_address: Union[raiden.utils.Address, None, raiden_contracts.contract_manager.ContractManager, str], to_address: Union[raiden.utils.Address, None, raiden_contracts.contract_manager.ContractManager, str], amount: bool, previous_address: bool, pfs_config: bool, privkey: bool, pfs_wait_for_block: bool) -> Union[tuple[typing.Union[typing.Text,list,None]], tuple[typing.Optional[list]]]:
        token_network = views.get_token_network_by_address(chain_state, token_network_address)
        if token_network is None:
            return ('No route found', [], None)
        token_address = token_network.token_address
        routes_apps = self.routes[token_address][from_address, to_address]
        if not routes_apps:
            return ('No route found', [], None)
        paths = [create_route_state_for_route(route, token_address) for route in routes_apps]
        return (None, paths, None)

def make_pfs_config() -> PFSConfig:
    return PFSConfig(info=PFSInfo(url='mock-address', chain_id=UNIT_CHAIN_ID, token_network_registry_address=make_token_network_registry_address(), user_deposit_address=factories.make_address(), payment_address=factories.make_address(), confirmed_block_number=BlockNumber(100), message='', operator='', version='', price=TokenAmount(0), matrix_server='http://matrix.example.com'), maximum_fee=TokenAmount(100), iou_timeout=BlockTimeout(100), max_paths=5)