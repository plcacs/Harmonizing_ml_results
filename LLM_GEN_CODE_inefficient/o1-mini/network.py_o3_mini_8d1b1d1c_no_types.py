""" Utilities to set-up a Raiden network. """
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import gevent
import structlog
from web3 import Web3
from raiden import waiting
from raiden.constants import BLOCK_ID_LATEST, GENESIS_BLOCK_NUMBER, Environment, RoutingMode
from raiden.exceptions import PFSReturnedError
from raiden.network.pathfinding import PFSProxy
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.proxies.secret_registry import SecretRegistry
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.proxies.token_network_registry import TokenNetworkRegistry
from raiden.network.rpc.client import JSONRPCClient
from raiden.raiden_event_handler import RaidenEventHandler
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, DEFAULT_RETRY_TIMEOUT, BlockchainConfig, CapabilitiesConfig, MatrixTransportConfig, MediationFeeConfig, RaidenConfig, RestApiConfig, ServiceConfig
from raiden.tests.utils.app import database_from_privatekey
from raiden.tests.utils.factories import UNIT_CHAIN_ID
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transport import ParsedURL, TestMatrixTransport
from raiden.transfer import views
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.views import get_channelstate_by_canonical_identifier, get_channelstate_by_token_network_and_partner, state_from_raiden
from raiden.ui.app import start_api_server
from raiden.ui.startup import RaidenBundle, ServicesBundle
from raiden.utils.formatting import to_checksum_address, to_hex_address
from raiden.utils.typing import Address, BlockIdentifier, BlockNumber, BlockTimeout, ChainID, Host, Iterable as TypingIterable, Iterator as TypingIterator, List as TypingList, MonitoringServiceAddress, OneToNAddress, Optional as TypingOptional, Port, PrivateKey, SecretRegistryAddress, ServiceRegistryAddress, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, Tuple as TypingTuple, UserDepositAddress
from raiden.waiting import wait_for_token_network
from raiden_contracts.contract_manager import ContractManager
AppChannels = Iterable[Tuple[RaidenService, RaidenService]]
log = structlog.get_logger(__name__)
CHAIN: Any = object()


@dataclass
class BlockchainServices:
    deploy_registry: TokenNetworkRegistry
    secret_registry: SecretRegistry
    service_registry: Optional[ServiceRegistry]
    proxy_manager: ProxyManager
    blockchain_services: List[ProxyManager]


def check_channel(app1, app2, token_network_address, settle_timeout,
    deposit_amount):
    channel_state1 = get_channelstate_by_token_network_and_partner(chain_state
        =state_from_raiden(app1), token_network_address=
        token_network_address, partner_address=app2.address)
    assert channel_state1, 'app1 does not have a channel with app2.'
    netcontract1 = app1.proxy_manager.payment_channel(channel_state=
        channel_state1, block_identifier=BLOCK_ID_LATEST)
    channel_state2 = get_channelstate_by_token_network_and_partner(chain_state
        =state_from_raiden(app2), token_network_address=
        token_network_address, partner_address=app1.address)
    assert channel_state2, 'app2 does not have a channel with app1.'
    netcontract2 = app2.proxy_manager.payment_channel(channel_state=
        channel_state2, block_identifier=BLOCK_ID_LATEST)
    assert settle_timeout == netcontract1.settle_timeout()
    assert settle_timeout == netcontract2.settle_timeout()
    if deposit_amount > 0:
        assert netcontract1.can_transfer(BLOCK_ID_LATEST)
        assert netcontract2.can_transfer(BLOCK_ID_LATEST)
    app1_details = netcontract1.detail(BLOCK_ID_LATEST)
    app2_details = netcontract2.detail(BLOCK_ID_LATEST)
    assert app1_details.participants_data.our_details.address == app2_details.participants_data.partner_details.address
    assert app1_details.participants_data.partner_details.address == app2_details.participants_data.our_details.address
    assert app1_details.participants_data.our_details.deposit == app2_details.participants_data.partner_details.deposit
    assert app1_details.participants_data.partner_details.deposit == app2_details.participants_data.our_details.deposit
    assert app1_details.chain_id == app2_details.chain_id
    assert app1_details.participants_data.our_details.deposit == deposit_amount
    assert app1_details.participants_data.partner_details.deposit == deposit_amount
    assert app2_details.participants_data.our_details.deposit == deposit_amount
    assert app2_details.participants_data.partner_details.deposit == deposit_amount
    assert app2_details.chain_id == UNIT_CHAIN_ID


def payment_channel_open_and_deposit(app0, app1, token_address, deposit,
    settle_timeout):
    """Open a new channel with app0 and app1 as participants"""
    assert token_address
    block_identifier: BlockIdentifier
    if app0.wal:
        block_identifier = views.get_confirmed_blockhash(app0)
    else:
        block_identifier = BLOCK_ID_LATEST
    token_network_address: TokenNetworkAddress = (app0.default_registry.
        get_token_network(token_address=token_address, block_identifier=
        block_identifier))
    assert token_network_address, 'request a channel for an unregistered token'
    token_network_proxy: TokenNetworkRegistry = (app0.proxy_manager.
        token_network(token_network_address, block_identifier=BLOCK_ID_LATEST))
    channel_details = token_network_proxy.new_netting_channel(partner=app1.
        address, settle_timeout=settle_timeout, given_block_identifier=
        block_identifier)
    channel_identifier: int = channel_details.channel_identifier
    assert channel_identifier
    if deposit != 0:
        for app, partner in [(app0, app1), (app1, app0)]:
            waiting.wait_for_newchannel(raiden=app,
                token_network_registry_address=app.default_registry.address,
                token_address=token_address, partner_address=partner.
                address, retry_timeout=0.5)
            chain_state = state_from_raiden(app)
            canonical_identifier: CanonicalIdentifier = CanonicalIdentifier(
                chain_identifier=chain_state.chain_id,
                token_network_address=token_network_proxy.address,
                channel_identifier=channel_identifier)
            channel_state = get_channelstate_by_canonical_identifier(
                chain_state=chain_state, canonical_identifier=
                canonical_identifier)
            assert channel_state, 'nodes dont share a channel'
            token = app.proxy_manager.token(token_address, BLOCK_ID_LATEST)
            payment_channel_proxy = app.proxy_manager.payment_channel(
                channel_state=channel_state, block_identifier=BLOCK_ID_LATEST)
            previous_balance: int = token.balance_of(app.address)
            assert previous_balance >= deposit
            payment_channel_proxy.approve_and_set_total_deposit(total_deposit
                =deposit, block_identifier=BLOCK_ID_LATEST)
            new_balance: int = token.balance_of(app.address)
            assert new_balance <= previous_balance - deposit
        check_channel(app0, app1, token_network_proxy.address,
            settle_timeout, deposit)


def create_all_channels_for_network(app_channels, token_addresses,
    channel_individual_deposit, channel_settle_timeout):
    greenlets: set[gevent.Greenlet] = set()
    for token_address in token_addresses:
        for app_pair in app_channels:
            greenlets.add(gevent.spawn(payment_channel_open_and_deposit,
                app_pair[0], app_pair[1], token_address,
                channel_individual_deposit, channel_settle_timeout))
    gevent.joinall(greenlets, raise_error=True)
    channels: List[Dict[str, Any]] = [{'app0': to_hex_address(app0.address),
        'app1': to_hex_address(app1.address), 'deposit':
        channel_individual_deposit, 'token_address': to_hex_address(
        token_address)} for (app0, app1), token_address in product(
        app_channels, token_addresses)]
    log.info('Test channels', channels=channels)


def network_with_minimum_channels(apps, channels_per_node):
    """Return the channels that should be created so that each app has at
    least `channels_per_node` with the other apps.

    Yields a two-tuple (app1, app2) that must be connected to respect
    `channels_per_node`. Any preexisting channels will be ignored, so the nodes
    might end up with more channels open than `channels_per_node`.
    """
    if channels_per_node > len(apps):
        raise ValueError("Can't create more channels than nodes")
    if len(apps) == 1:
        raise ValueError("Can't create channels with only one node")
    unconnected_apps: Dict[Address, List[RaidenService]] = {}
    channel_count: Dict[Address, int] = {}
    for curr_app in apps:
        all_apps = list(apps)
        all_apps.remove(curr_app)
        unconnected_apps[curr_app.address] = all_apps
        channel_count[curr_app.address] = 0
    for curr_app in sorted(apps, key=lambda app: app.address):
        available_apps: List[RaidenService] = unconnected_apps[curr_app.address
            ]
        while channel_count[curr_app.address] < channels_per_node:
            least_connect: RaidenService = sorted(available_apps, key=lambda
                app: channel_count[app.address])[0]
            channel_count[curr_app.address] += 1
            available_apps.remove(least_connect)
            channel_count[least_connect.address] += 1
            unconnected_apps[least_connect.address].remove(curr_app)
            yield curr_app, least_connect


def create_network_channels(raiden_apps, channels_per_node):
    app_channels: AppChannels
    num_nodes: int = len(raiden_apps)
    if channels_per_node is not CHAIN and channels_per_node > num_nodes:
        raise ValueError("Can't create more channels than nodes")
    if channels_per_node == 0:
        app_channels = []
    elif channels_per_node == CHAIN:
        app_channels = list(zip(raiden_apps[:-1], raiden_apps[1:]))
    else:
        app_channels = list(network_with_minimum_channels(raiden_apps,
            channels_per_node))
    return app_channels


def create_sequential_channels(raiden_apps, channels_per_node):
    """Create a fully connected network with `num_nodes`, the nodes are
    connected sequentially.

    Returns:
        A list of apps of size `num_nodes`, with the property that every
        sequential pair in the list has an open channel with `deposit` for each
        participant.
    """
    app_channels: AppChannels
    num_nodes: int = len(raiden_apps)
    if num_nodes < 2:
        raise ValueError('cannot create a network with less than two nodes')
    if channels_per_node not in (0, 1, 2, CHAIN):
        raise ValueError(
            'can only create networks with 0, 1, 2 or CHAIN channels')
    if channels_per_node == 0:
        app_channels = []
    elif channels_per_node == 1:
        if len(raiden_apps) % 2 != 0:
            raise ValueError('needs an even number of nodes')
        every_two = iter(raiden_apps)
        app_channels = list(zip(every_two, every_two))
    elif channels_per_node == 2:
        app_channels = list(zip(raiden_apps, raiden_apps[1:] + [raiden_apps
            [0]]))
    elif channels_per_node == CHAIN:
        app_channels = list(zip(raiden_apps[:-1], raiden_apps[1:]))
    return app_channels


def create_apps(chain_id, contracts_path, blockchain_services,
    token_network_registry_address, one_to_n_address,
    secret_registry_address, service_registry_address, user_deposit_address,
    monitoring_service_contract_address, reveal_timeout, settle_timeout,
    database_basedir, retry_interval_initial, retry_interval_max,
    retries_before_backoff, environment_type,
    unrecoverable_error_should_crash, local_matrix_url, routing_mode,
    blockchain_query_interval, resolver_ports, enable_rest_api,
    port_generator, capabilities_config):
    """Create the apps."""
    apps: List[RaidenService] = []
    for idx, proxy_manager in enumerate(blockchain_services):
        database_path: str = database_from_privatekey(base_dir=
            database_basedir, app_number=idx)
        assert len(resolver_ports) > idx
        resolver_port: TypingOptional[int] = resolver_ports[idx]
        config: RaidenConfig = RaidenConfig(chain_id=chain_id,
            environment_type=environment_type,
            unrecoverable_error_should_crash=
            unrecoverable_error_should_crash, reveal_timeout=reveal_timeout,
            settle_timeout=settle_timeout, contracts_path=contracts_path,
            database_path=database_path, blockchain=BlockchainConfig(
            confirmation_blocks=DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS,
            query_interval=blockchain_query_interval), mediation_fees=
            MediationFeeConfig(), services=ServiceConfig(monitoring_enabled
            =False), rest_api=RestApiConfig(rest_api_enabled=
            enable_rest_api, host=Host('localhost'), port=next(
            port_generator)), console=False, transport_type='matrix')
        config.transport.capabilities_config = capabilities_config
        if local_matrix_url is not None:
            config.transport = MatrixTransportConfig(retries_before_backoff
                =retries_before_backoff, retry_interval_initial=
                retry_interval_initial, retry_interval_max=
                retry_interval_max, server=local_matrix_url,
                available_servers=[], capabilities_config=capabilities_config)
        assert config.transport.capabilities_config is not None
        if resolver_port is not None:
            config.resolver_endpoint = f'http://localhost:{resolver_port}'
        registry: TokenNetworkRegistry = proxy_manager.token_network_registry(
            token_network_registry_address, block_identifier=BLOCK_ID_LATEST)
        secret_registry: SecretRegistry = proxy_manager.secret_registry(
            secret_registry_address, block_identifier=BLOCK_ID_LATEST)
        services_bundle: TypingOptional[ServicesBundle] = None
        if user_deposit_address:
            user_deposit = proxy_manager.user_deposit(user_deposit_address,
                block_identifier=BLOCK_ID_LATEST)
            service_registry: TypingOptional[ServiceRegistry] = None
            if service_registry_address:
                service_registry = proxy_manager.service_registry(
                    service_registry_address, block_identifier=BLOCK_ID_LATEST)
            monitoring_service: TypingOptional[Any] = None
            if monitoring_service_contract_address:
                monitoring_service = proxy_manager.monitoring_service(
                    monitoring_service_contract_address, block_identifier=
                    BLOCK_ID_LATEST)
            one_to_n: TypingOptional[Any] = None
            if one_to_n_address:
                one_to_n = proxy_manager.one_to_n(one_to_n_address,
                    block_identifier=BLOCK_ID_LATEST)
            services_bundle = ServicesBundle(user_deposit, service_registry,
                monitoring_service, one_to_n)
        assert config.transport.capabilities_config is not None
        transport: TestMatrixTransport = TestMatrixTransport(config=config.
            transport, environment=environment_type)
        raiden_event_handler: RaidenEventHandler = RaidenEventHandler()
        hold_handler: HoldRaidenEventHandler = HoldRaidenEventHandler(
            raiden_event_handler)
        message_handler: WaitForMessage = WaitForMessage()
        api_server: TypingOptional[Any] = None
        if enable_rest_api:
            api_server = start_api_server(rpc_client=proxy_manager.client,
                config=config.rest_api, eth_rpc_endpoint='bla')
        app: RaidenService = RaidenService(config=config, rpc_client=
            proxy_manager.client, proxy_manager=proxy_manager,
            query_start_block=BlockNumber(0), raiden_bundle=RaidenBundle(
            registry, secret_registry), services_bundle=services_bundle,
            transport=transport, raiden_event_handler=hold_handler,
            message_handler=message_handler, routing_mode=routing_mode,
            api_server=api_server, pfs_proxy=SimplePFSProxy(apps))
        apps.append(app)
    return apps


class SimplePFSProxy(PFSProxy):

    def __init__(self, services):
        self._services: List[RaidenService] = services

    def query_address_metadata(self, address):
        for service in self._services:
            if service.address == address:
                return service.transport.address_metadata
        raise PFSReturnedError(message='Address not found', error_code=404,
            error_details={})

    def set_services(self, services):
        self._services = services


def parallel_start_apps(raiden_apps):
    """Start all the raiden apps in parallel."""
    start_tasks: set[gevent.Greenlet] = set()
    for app in raiden_apps:
        greenlet: gevent.Greenlet = gevent.spawn(app.start)
        greenlet.name = (
            f'Fixture:raiden_network node:{to_checksum_address(app.address)}')
        start_tasks.add(greenlet)
    gevent.joinall(start_tasks, raise_error=True)
    addresses_in_order: Dict[int, str] = {pos: to_hex_address(app.address) for
        pos, app in enumerate(raiden_apps)}
    log.info('Raiden Apps started', addresses_in_order=addresses_in_order)


def jsonrpc_services(proxy_manager, private_keys, secret_registry_address,
    service_registry_address, token_network_registry_address, web3,
    contract_manager):
    block_identifier: BlockIdentifier = BLOCK_ID_LATEST
    secret_registry: SecretRegistry = proxy_manager.secret_registry(
        secret_registry_address, block_identifier=block_identifier)
    service_registry: TypingOptional[ServiceRegistry] = None
    if service_registry_address:
        service_registry = proxy_manager.service_registry(
            service_registry_address, block_identifier=block_identifier)
    deploy_registry: TokenNetworkRegistry = (proxy_manager.
        token_network_registry(token_network_registry_address,
        block_identifier=block_identifier))
    blockchain_services: List[ProxyManager] = []
    for privkey in private_keys:
        rpc_client: JSONRPCClient = JSONRPCClient(web3, privkey)
        new_proxy_manager: ProxyManager = ProxyManager(rpc_client=
            rpc_client, contract_manager=contract_manager, metadata=
            ProxyManagerMetadata(token_network_registry_deployed_at=
            GENESIS_BLOCK_NUMBER, filters_start_at=GENESIS_BLOCK_NUMBER))
        blockchain_services.append(new_proxy_manager)
    return BlockchainServices(deploy_registry=deploy_registry,
        secret_registry=secret_registry, service_registry=service_registry,
        proxy_manager=proxy_manager, blockchain_services=blockchain_services)


def wait_for_alarm_start(raiden_apps, retry_timeout=DEFAULT_RETRY_TIMEOUT):
    """Wait until all Alarm tasks start & set up the last_block"""
    apps: List[RaidenService] = list(raiden_apps)
    while apps:
        app: RaidenService = apps[-1]
        if app.alarm.known_block_number is None:
            gevent.sleep(retry_timeout)
        else:
            apps.pop()


def wait_for_usable_channel(raiden, partner_address,
    token_network_registry_address, token_address, our_deposit,
    partner_deposit, retry_timeout=DEFAULT_RETRY_TIMEOUT):
    """Wait until the channel from app0 to app1 is usable.

    The channel and the deposits are registered, and the partner network state
    is reachable.
    """
    waiting.wait_for_newchannel(raiden=raiden,
        token_network_registry_address=token_network_registry_address,
        token_address=token_address, partner_address=partner_address,
        retry_timeout=retry_timeout)
    waiting.wait_for_participant_deposit(raiden=raiden,
        token_network_registry_address=token_network_registry_address,
        token_address=token_address, partner_address=partner_address,
        target_address=raiden.address, target_balance=our_deposit,
        retry_timeout=retry_timeout)
    waiting.wait_for_participant_deposit(raiden=raiden,
        token_network_registry_address=token_network_registry_address,
        token_address=token_address, partner_address=partner_address,
        target_address=partner_address, target_balance=partner_deposit,
        retry_timeout=retry_timeout)


def wait_for_token_networks(raiden_apps, token_network_registry_address,
    token_addresses, retry_timeout=DEFAULT_RETRY_TIMEOUT):
    for token_address in token_addresses:
        for app in raiden_apps:
            wait_for_token_network(app, token_network_registry_address,
                token_address, retry_timeout)


def wait_for_channels(app_channels, token_network_registry_address,
    token_addresses, deposit, retry_timeout=DEFAULT_RETRY_TIMEOUT):
    """Wait until all channels are usable from both directions."""
    for app0, app1 in app_channels:
        for token_address in token_addresses:
            wait_for_usable_channel(raiden=app0, partner_address=app1.
                address, token_network_registry_address=
                token_network_registry_address, token_address=token_address,
                our_deposit=deposit, partner_deposit=deposit, retry_timeout
                =retry_timeout)
            wait_for_usable_channel(raiden=app1, partner_address=app0.
                address, token_network_registry_address=
                token_network_registry_address, token_address=token_address,
                our_deposit=deposit, partner_deposit=deposit, retry_timeout
                =retry_timeout)
