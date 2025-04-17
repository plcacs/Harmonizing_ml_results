import os
import random
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Set, Tuple, Optional, cast
from uuid import UUID
import click
import filelock
import gevent
import structlog
from eth_utils import to_hex
from gevent import Greenlet
from gevent.event import AsyncResult, Event
from web3.types import BlockData
from raiden import routing
from raiden.api.objects import Notification
from raiden.api.python import RaidenAPI
from raiden.api.rest import APIServer, RestAPI
from raiden.blockchain.decode import blockchainevent_to_statechange
from raiden.blockchain.events import BlockchainEvents, DecodedEvent
from raiden.blockchain.filters import RaidenContractFilter
from raiden.constants import ABSENT_SECRET, BLOCK_ID_LATEST, GENESIS_BLOCK_NUMBER, SECRET_LENGTH, SNAPSHOT_STATE_CHANGES_COUNT, Environment, RoutingMode
from raiden.exceptions import BrokenPreconditionError, InvalidDBData, InvalidSecret, InvalidSecretHash, InvalidSettleTimeout, PaymentConflict, RaidenRecoverableError, RaidenUnrecoverableError, SerializationError
from raiden.message_handler import MessageHandler
from raiden.messages.abstract import Message, SignedMessage
from raiden.messages.encode import message_from_sendevent
from raiden.network.pathfinding import PFSProxy
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.network.proxies.secret_registry import SecretRegistry
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.proxies.token_network_registry import TokenNetworkRegistry
from raiden.network.proxies.user_deposit import UserDeposit
from raiden.network.rpc.client import JSONRPCClient
from raiden.network.transport import populate_services_addresses
from raiden.network.transport.matrix.transport import MatrixTransport, MessagesQueue
from raiden.raiden_event_handler import EventHandler
from raiden.services import send_pfs_update, update_monitoring_service_from_balance_proof
from raiden.settings import RaidenConfig
from raiden.storage import sqlite, wal
from raiden.storage.serialization import DictSerializer, JSONSerializer
from raiden.storage.sqlite import HIGH_STATECHANGE_ULID, Range
from raiden.storage.wal import WriteAheadLog
from raiden.tasks import AlarmTask
from raiden.transfer import node, views
from raiden.transfer.architecture import BalanceProofSignedState, ContractSendEvent, Event as RaidenEvent, StateChange
from raiden.transfer.channel import get_capacity
from raiden.transfer.events import EventPaymentSentFailed, EventPaymentSentSuccess, EventWrapper, RequestMetadata, SendWithdrawExpired, SendWithdrawRequest
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import EventRouteFailed, SendLockedTransfer, SendSecretRequest, SendUnlock
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState, calculate_imbalance_fees
from raiden.transfer.mediated_transfer.state import TransferDescriptionWithSecretState
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ReceiveLockExpired, ReceiveTransferCancelRoute, ReceiveTransferRefund
from raiden.transfer.mediated_transfer.tasks import InitiatorTask
from raiden.transfer.state import ChainState, RouteState, TokenNetworkRegistryState
from raiden.transfer.state_change import ActionChannelSetRevealTimeout, ActionChannelWithdraw, BalanceProofStateChange, Block, ContractReceiveChannelDeposit, ReceiveUnlock, ReceiveWithdrawExpired, ReceiveWithdrawRequest
from raiden.ui.startup import RaidenBundle, ServicesBundle
from raiden.utils.formatting import lpex, to_checksum_address
from raiden.utils.gevent import spawn_named
from raiden.utils.logging import redact_secret
from raiden.utils.runnable import Runnable
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.signer import LocalSigner, Signer
from raiden.utils.transfers import random_secret
from raiden.utils.typing import MYPY_ANNOTATION, Address, AddressMetadata, BlockNumber, BlockTimeout, InitiatorAddress, MonitoringServiceAddress, OneToNAddress, PaymentAmount, PaymentID, PrivateKey, Secret, SecretHash, SecretRegistryAddress, TargetAddress, TokenNetworkAddress, WithdrawAmount, typecheck
from raiden.utils.upgrades import UpgradeManager
from raiden_contracts.constants import ChannelEvent
from raiden_contracts.contract_manager import ContractManager
log = structlog.get_logger(__name__)
StatusesDict = Dict[TargetAddress, Dict[PaymentID, 'PaymentStatus']]
PFS_UPDATE_CAPACITY_STATE_CHANGES = (ContractReceiveChannelDeposit, ReceiveUnlock, ReceiveWithdrawRequest, ReceiveWithdrawExpired, ReceiveTransferCancelRoute, ReceiveLockExpired, ReceiveTransferRefund)
PFS_UPDATE_CAPACITY_EVENTS = (SendUnlock, SendLockedTransfer, SendWithdrawRequest, SendWithdrawExpired)
PFS_UPDATE_FEE_STATE_CHANGES = (ContractReceiveChannelDeposit, ReceiveWithdrawRequest, ReceiveWithdrawExpired)
PFS_UPDATE_FEE_EVENTS = (SendWithdrawRequest, SendWithdrawExpired)
assert not set(PFS_UPDATE_FEE_STATE_CHANGES) - set(PFS_UPDATE_CAPACITY_STATE_CHANGES), 'No fee updates without capacity updates possible'
assert not set(PFS_UPDATE_FEE_EVENTS) - set(PFS_UPDATE_CAPACITY_EVENTS), 'No fee updates without capacity updates possible'

def initiator_init(raiden, transfer_identifier, transfer_amount, transfer_secret, transfer_secrethash, token_network_address, target_address, lock_timeout=None, route_states=None):
    transfer_state = TransferDescriptionWithSecretState(token_network_registry_address=raiden.default_registry.address, payment_identifier=transfer_identifier, amount=transfer_amount, token_network_address=token_network_address, initiator=InitiatorAddress(raiden.address), target=target_address, secret=transfer_secret, secrethash=transfer_secrethash, lock_timeout=lock_timeout)
    error_msg = None
    if route_states is None:
        our_address_metadata = raiden.transport.address_metadata
        msg = 'Transport is not initialized with raiden-service'
        assert our_address_metadata is not None, msg
        error_msg, route_states, feedback_token = routing.get_best_routes(chain_state=views.state_from_raiden(raiden), token_network_address=token_network_address, one_to_n_address=raiden.default_one_to_n_address, from_address=InitiatorAddress(raiden.address), to_address=target_address, amount=transfer_amount, previous_address=None, privkey=raiden.privkey, our_address_metadata=our_address_metadata, pfs_proxy=raiden.pfs_proxy)
        if feedback_token is not None:
            for route_state in route_states:
                raiden.route_to_feedback_token[tuple(route_state.route)] = feedback_token
    return (error_msg, ActionInitInitiator(transfer_state, route_states))

def smart_contract_filters_from_node_state(chain_state, secret_registry_address, service_registry):
    token_network_registries = chain_state.identifiers_to_tokennetworkregistries.values()
    token_networks = [tn for tnr in token_network_registries for tn in tnr.token_network_list]
    channels_of_token_network = {tn.address: set(tn.channelidentifiers_to_channels.keys()) for tn in token_networks if tn.channelidentifiers_to_channels}
    return RaidenContractFilter(secret_registry_address=secret_registry_address, token_network_registry_addresses={tnr.address for tnr in token_network_registries}, token_network_addresses={tn.address for tn in token_networks}, channels_of_token_network=channels_of_token_network, ignore_secret_registry_until_channel_found=not channels_of_token_network, service_registry=service_registry)

class PaymentStatus(NamedTuple):
    """Value type for RaidenService.targets_to_identifiers_to_statuses.

    Contains the necessary information to tell conflicting transfers from
    retries as well as the status of a transfer that is retried.
    """
    payment_identifier: PaymentID
    amount: PaymentAmount
    token_network_address: TokenNetworkAddress
    payment_done: AsyncResult
    lock_timeout: Optional[BlockTimeout]

    def matches(self, token_network_address, amount):
        return token_network_address == self.token_network_address and amount == self.amount

class SyncTimeout:
    """Helper to determine if the sync should halt or continue.

    The goal of this helper is to stop syncing before the block
    `current_confirmed_head` is pruned, otherwise JSON-RPC requests will start
    to fail.
    """

    def __init__(self, current_confirmed_head, timeout):
        self.sync_start = time.monotonic()
        self.timeout = timeout
        self.current_confirmed_head = current_confirmed_head

    def time_elapsed(self):
        delta = time.monotonic() - self.sync_start
        return delta

    def should_continue(self, last_fetched_block):
        has_time = self.timeout >= self.time_elapsed()
        has_blocks_unsynched = self.current_confirmed_head > last_fetched_block
        return has_time and has_blocks_unsynched

class SynchronizationState(Enum):
    FULLY_SYNCED = 'fully_synced'
    PARTIALLY_SYNCED = 'partially_synced'

class RaidenService(Runnable):
    """A Raiden node."""

    def __init__(self, rpc_client, proxy_manager, query_start_block, raiden_bundle, services_bundle, transport, raiden_event_handler, message_handler, routing_mode, config, api_server=None, pfs_proxy=None):
        super().__init__()
        settlement_timeout_min = raiden_bundle.token_network_registry.settlement_timeout_min(BLOCK_ID_LATEST)
        settlement_timeout_max = raiden_bundle.token_network_registry.settlement_timeout_max(BLOCK_ID_LATEST)
        invalid_settle_timeout = config.settle_timeout < settlement_timeout_min or config.settle_timeout > settlement_timeout_max or config.settle_timeout < config.reveal_timeout * 2
        if invalid_settle_timeout:
            contract = to_checksum_address(raiden_bundle.token_network_registry.address)
            raise InvalidSettleTimeout(f'Settlement timeout for Registry contract {contract} must be in range [{settlement_timeout_min}, {settlement_timeout_max}], is {config.settle_timeout}')
        self.targets_to_identifiers_to_statuses: StatusesDict = defaultdict(dict)
        one_to_n_address = None
        monitoring_service_address = None
        service_registry: Optional[ServiceRegistry] = None
        user_deposit: Optional[UserDeposit] = None
        if services_bundle:
            if services_bundle.one_to_n:
                one_to_n_address = services_bundle.one_to_n.address
            if services_bundle.monitoring_service:
                monitoring_service_address = services_bundle.monitoring_service.address
            service_registry = services_bundle.service_registry
            user_deposit = services_bundle.user_deposit
        self.rpc_client: JSONRPCClient = rpc_client
        self.proxy_manager: ProxyManager = proxy_manager
        self.default_registry: TokenNetworkRegistry = raiden_bundle.token_network_registry
        self.query_start_block = query_start_block
        self.default_services_bundle = services_bundle
        self.default_one_to_n_address: Optional[OneToNAddress] = one_to_n_address
        self.default_secret_registry: SecretRegistry = raiden_bundle.secret_registry
        self.default_service_registry = service_registry
        self.default_user_deposit: Optional[UserDeposit] = user_deposit
        self.default_msc_address: Optional[MonitoringServiceAddress] = monitoring_service_address
        self.routing_mode: RoutingMode = routing_mode
        self.config: RaidenConfig = config
        self.notifications: Dict = {}
        self.signer: Signer = LocalSigner(self.rpc_client.privkey)
        self.address: Address = self.signer.address
        self.transport: MatrixTransport = transport
        self.alarm = AlarmTask(proxy_manager=proxy_manager, sleep_time=self.config.blockchain.query_interval)
        self.raiden_event_handler = raiden_event_handler
        self.message_handler = message_handler
        self.blockchain_events: Optional[BlockchainEvents] = None
        self.api_server: Optional[APIServer] = api_server
        self.raiden_api: Optional[RaidenAPI] = None
        self.rest_api: Optional[RestAPI] = None
        if api_server is not None:
            self.raiden_api = RaidenAPI(self)
            self.rest_api = api_server.rest_api
        self.stop_event = Event()
        self.stop_event.set()
        self.greenlets: List[Greenlet] = []
        self.last_log_time = time.monotonic()
        self.last_log_block = BlockNumber(0)
        self.contract_manager: ContractManager = ContractManager(config.contracts_path)
        self.wal: Optional[WriteAheadLog] = None
        self.db_lock: Optional[filelock.UnixFileLock] = None
        if pfs_proxy is None:
            assert config.pfs_config is not None, 'must not be None'
            pfs_proxy = PFSProxy(config.pfs_config)
        self.pfs_proxy = pfs_proxy
        if self.config.database_path != ':memory:':
            database_dir = os.path.dirname(config.database_path)
            os.makedirs(database_dir, exist_ok=True)
            self.database_dir: Optional[str] = database_dir
            lock_file = os.path.join(self.database_dir, '.lock')
            self.db_lock = filelock.FileLock(lock_file)
        else:
            self.database_dir = None
            self.serialization_file = None
            self.db_lock = None
        self.payment_identifier_lock = gevent.lock.Semaphore()
        self.route_to_feedback_token: Dict[Tuple[Address, ...], UUID] = {}
        self.ready_to_process_events = False
        self.state_change_qty_snapshot = 0
        self.state_change_qty = 0

    def start(self):
        """Start the node synchronously. Raises directly if anything went wrong on startup"""
        assert self.stop_event.ready(), f'Node already started. node:{self!r}'
        self.stop_event.clear()
        self.greenlets = []
        self.ready_to_process_events = False
        self._initialize_wal()
        self._synchronize_with_blockchain()
        chain_state = views.state_from_raiden(self)
        self._initialize_payment_statuses(chain_state)
        self._initialize_transactions_queues(chain_state)
        self._initialize_messages_queues(chain_state)
        self._initialize_channel_fees()
        self._initialize_monitoring_services_queue(chain_state)
        self._initialize_ready_to_process_events()
        self.alarm.greenlet.link_exception(self.on_error)
        self.transport.greenlet.link_exception(self.on_error)
        if self.api_server is not None:
            self.api_server.greenlet.link_exception(self.on_error)
        self._start_transport()
        self._start_alarm_task()
        log.debug('Raiden Service started', node=to_checksum_address(self.address))
        super().start()
        self._set_rest_api_service_available()

    def _run(self, *args: Any, **kwargs: Any):
        """Busy-wait on long-lived subtasks/greenlets, re-raise if any error occurs"""
        self.greenlet.name = f'RaidenService._run node:{to_checksum_address(self.address)}'
        try:
            self.stop_event.wait()
        except gevent.GreenletExit:
            self.stop_event.set()
            gevent.killall([self.alarm.greenlet, self.transport.greenlet])
            raise

    def stop(self):
        """Stop the node gracefully. Raise if any stop-time error occurred on any subtask"""
        if self.stop_event.ready():
            return
        self.stop_event.set()
        if self.api_server is not None:
            self.api_server.stop()
            self.api_server.greenlet.join()
        self.alarm.stop()
        self.alarm.greenlet.join()
        self.transport.stop()
        self.transport.greenlet.join()
        assert self.blockchain_events, f'The blockchain_events has to be set by the start. node:{self!r}'
        self.blockchain_events.stop()
        assert self.wal, f'The Service must have been started before it can be stopped. node:{self!r}'
        self.wal.storage.close()
        self.wal = None
        if self.db_lock is not None:
            self.db_lock.release()
        log.debug('Raiden Service stopped', node=to_checksum_address(self.address))

    def add_notification(self, notification, log_opts=None, click_opts=None):
        log_opts = log_opts or {}
        click_opts = click_opts or {}
        log.info(notification.summary, **log_opts)
        click.secho(notification.body, **click_opts)
        self.notifications[notification.id] = notification

    @property
    def confirmation_blocks(self):
        return self.config.blockchain.confirmation_blocks

    @property
    def privkey(self):
        return self.rpc_client.privkey

    def add_pending_greenlet(self, greenlet):
        """Ensures an error on the passed greenlet crashes self/main greenlet."""

        def remove(_):
            self.greenlets.remove(greenlet)
        self.greenlets.append(greenlet)
        greenlet.link_exception(self.on_error)
        greenlet.link_value(remove)

    def __repr__(self):
        return f'<{self.__class__.__name__} node:{to_checksum_address(self.address)}>'

    def _start_transport(self):
        """Initialize the transport and related facilities.

        Note:
            The node has first to `_synchronize_with_blockchain` before
            starting the transport. This synchronization includes the on-chain
            channel state and is necessary to reject new messages for closed
            channels.
        """
        assert self.ready_to_process_events, f'Event processing disabled. node:{self!r}'
        msg = "`self.blockchain_events` is `None`. Seems like `_synchronize_with_blockchain` wasn't called before `_start_transport`."
        assert self.blockchain_events is not None, msg
        if self.default_service_registry is not None:
            populate_services_addresses(self.transport, self.default_service_registry, BLOCK_ID_LATEST)
        self.transport.start(raiden_service=self, prev_auth_data=None)

    def _make_initial_state(self):
        last_log_block_number = self.query_start_block
        last_log_block_hash = self.rpc_client.blockhash_from_blocknumber(last_log_block_number)
        initial_state = ChainState(pseudo_random_generator=random.Random(), block_number=last_log_block_number, block_hash=last_log_block_hash, our_address=self.address, chain_id=self.rpc_client.chain_id)
        token_network_registry_address = self.default_registry.address
        token_network_registry = TokenNetworkRegistryState(token_network_registry_address, [])
        initial_state.identifiers_to_tokennetworkregistries[token_network_registry_address] = token_network_registry
        return initial_state

    def _initialize_wal(self):
        if self.database_dir is not None:
            try:
                assert self.db_lock is not None, 'If a database_dir is present, a lock for the database has to exist'
                self.db_lock.acquire(timeout=0)
                assert self.db_lock.is_locked, f'Database not locked. node:{self!r}'
            except (filelock.Timeout, AssertionError) as ex:
                raise RaidenUnrecoverableError(f'Could not aquire database lock. Maybe a Raiden node for this account ({to_checksum_address(self.address)}) is already running?') from ex
        self.maybe_upgrade_db()
        storage = sqlite.SerializedSQLiteStorage(database_path=self.config.database_path, serializer=JSONSerializer())
        storage.update_version()
        storage.log_run()
        try:
            initial_state = self._make_initial_state()
            state_snapshot, state_change_start, state_change_qty_snapshot = wal.restore_or_init_snapshot(storage=storage, node_address=self.address, initial_state=initial_state)
            state, state_change_qty_unapplied = wal.replay_state_changes(node_address=self.address, state=state_snapshot, state_change_range=Range(state_change_start, HIGH_STATECHANGE_ULID), storage=storage, transition_function=node.state_transition)
        except SerializationError:
            raise RaidenUnrecoverableError('Could not restore state. It seems like the existing database is incompatible with the current version of Raiden. Consider using a stable version of the Raiden client.')
        if state_change_qty_snapshot == 0:
            print('This is the first time Raiden is being used with this address. Processing all the events may take some time. Please wait ...')
        self.state_change_qty_snapshot = state_change_qty_snapshot
        self.state_change_qty = state_change_qty_snapshot + state_change_qty_unapplied
        msg = 'The state must be a ChainState instance.'
        assert isinstance(state, ChainState), msg
        self.wal = WriteAheadLog(state, storage, node.state_transition)
        last_log_block_number = views.block_number(self.wal.get_current_state())
        log.debug('Querying blockchain from block', last_restored_block=last_log_block_number, node=to_checksum_address(self.address))
        known_networks = views.get_token_network_registry_address(views.state_from_raiden(self))
        if known_networks and self.default_registry.address not in known_networks:
            configured_registry = to_checksum_address(self.default_registry.address)
            known_registries = lpex(known_networks)
            raise RuntimeError(f'Token network address mismatch.\nRaiden is configured to use the smart contract {configured_registry}, which conflicts with the current known smart contracts {known_registries}')

    def _log_sync_progress(self, polled_block_number, target_block):
        """Print a message if there are many blocks to be fetched, or if the
        time in-between polls is high.
        """
        now = time.monotonic()
        blocks_until_target = target_block - polled_block_number
        polled_block_count = polled_block_number - self.last_log_block
        elapsed = now - self.last_log_time
        if blocks_until_target > 100 or elapsed > 15.0:
            log.info('Synchronizing blockchain events', remaining_blocks_to_sync=blocks_until_target, blocks_per_second=polled_block_count / elapsed, to_block=target_block, elapsed=elapsed)
            self.last_log_time = time.monotonic()
            self.last_log_block = polled_block_number

    def _synchronize_with_blockchain(self):
        """Prepares the alarm task callback and synchronize with the blockchain
        since the last run.

         Notes about setup order:
         - The filters must be polled after the node state has been primed,
           otherwise the state changes won't have effect.
         - The synchronization must be done before the transport is started, to
           reject messages for closed/settled channels.
        """
        msg = f'Transport must not be started before the node has synchronized with the blockchain, otherwise the node may accept transfers to a closed channel. node:{self!r}'
        assert not self.transport, msg
        assert self.wal, f'The database must have been initialized. node:{self!r}'
        chain_state = views.state_from_raiden(self)
        last_block_number = views.block_number(chain_state)
        event_filter = smart_contract_filters_from_node_state(chain_state, self.default_secret_registry.address, self.default_service_registry)
        log.debug('initial filter', event_filter=event_filter, node=self.address)
        blockchain_events = BlockchainEvents(web3=self.rpc_client.web3, chain_id=chain_state.chain_id, contract_manager=self.contract_manager, last_fetched_block=last_block_number, event_filter=event_filter, block_batch_size_config=self.config.blockchain.block_batch_size_config, node_address=self.address)
        blockchain_events.register_listener(self._blockchain_event_listener)
        self.last_log_block = last_block_number
        self.last_log_time = time.monotonic()
        self.blockchain_events = blockchain_events
        synchronization_state = SynchronizationState.PARTIALLY_SYNCED
        while synchronization_state is SynchronizationState.PARTIALLY_SYNCED:
            latest_block = self.rpc_client.get_block(block_identifier=BLOCK_ID_LATEST)
            synchronization_state = self._best_effort_synchronize(latest_block)
        self.alarm.register_callback(self._best_effort_synchronize)

    def _blockchain_event_listener(self, events):
        if not self.transport.started:
            return
        for event in events:
            args = event.event_data['args']
            if event.event_data['event'] == ChannelEvent.OPENED:
                other = args['participant1'] if args['participant1'] != self.address else args['participant2']
                self.transport.health_check_web_rtc(other)

    def _start_alarm_task(self):
        """Start the alarm task.

        Note:
            The alarm task must be started only when processing events is
            allowed, otherwise side-effects of blockchain events will be
            ignored.
        """
        assert self.ready_to_process_events, f'Event processing disabled. node:{self!r}'
        self.alarm.start()

    def _set_rest_api_service_available(self):
        if self.raiden_api:
            assert self.rest_api, 'api enabled in config but self.rest_api not initialized'
            self.rest_api.raiden_api = self.raiden_api
            print('Synchronization complete, REST API services now available.')

    def _initialize_ready_to_process_events(self):
        """Mark the node as ready to start processing raiden events that may
        send messages or transactions.

        This flag /must/ be set to true before the both  transport and the
        alarm are started.
        """
        msg = f'The transport must not be initialized before the `ready_to_process_events` flag is set, since this is a requirement for the alarm task and the alarm task should be started before the transport to avoid race conditions. node:{self!r}'
        assert not self.transport, msg
        msg = f'Alarm task must not be started before the `ready_to_process_events` flag is set, otherwise events may be missed. node:{self!r}'
        assert not self.alarm, msg
        self.ready_to_process_events = True

    def get_block_number(self):
        assert self.wal, f'WAL object not yet initialized. node:{self!r}'
        return views.block_number(self.wal.get_current_state())

    def on_messages(self, messages):
        self.message_handler.on_messages(self, messages)

    def handle_and_track_state_changes(self, state_changes):
        """Dispatch the state change and does not handle the exceptions.

        When the method is used the exceptions are tracked and re-raised in the
        raiden service thread.
        """
        if len(state_changes) == 0:
            return
        for greenlet in self.handle_state_changes(state_changes):
            self.add_pending_g