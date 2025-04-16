# pylint: disable=too-many-lines
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
from raiden.constants import (
    ABSENT_SECRET,
    BLOCK_ID_LATEST,
    GENESIS_BLOCK_NUMBER,
    SECRET_LENGTH,
    SNAPSHOT_STATE_CHANGES_COUNT,
    Environment,
    RoutingMode,
)
from raiden.exceptions import (
    BrokenPreconditionError,
    InvalidDBData,
    InvalidSecret,
    InvalidSecretHash,
    InvalidSettleTimeout,
    PaymentConflict,
    RaidenRecoverableError,
    RaidenUnrecoverableError,
    SerializationError,
)
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
from raiden.transfer.architecture import (
    BalanceProofSignedState,
    ContractSendEvent,
    Event as RaidenEvent,
    StateChange,
)
from raiden.transfer.channel import get_capacity
from raiden.transfer.events import (
    EventPaymentSentFailed,
    EventPaymentSentSuccess,
    EventWrapper,
    RequestMetadata,
    SendWithdrawExpired,
    SendWithdrawRequest,
)
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import (
    EventRouteFailed,
    SendLockedTransfer,
    SendSecretRequest,
    SendUnlock,
)
from raiden.transfer.mediated_transfer.mediation_fee import (
    FeeScheduleState,
    calculate_imbalance_fees,
)
from raiden.transfer.mediated_transfer.state import TransferDescriptionWithSecretState
from raiden.transfer.mediated_transfer.state_change import (
    ActionInitInitiator,
    ReceiveLockExpired,
    ReceiveTransferCancelRoute,
    ReceiveTransferRefund,
)
from raiden.transfer.mediated_transfer.tasks import InitiatorTask
from raiden.transfer.state import ChainState, RouteState, TokenNetworkRegistryState
from raiden.transfer.state_change import (
    ActionChannelSetRevealTimeout,
    ActionChannelWithdraw,
    BalanceProofStateChange,
    Block,
    ContractReceiveChannelDeposit,
    ReceiveUnlock,
    ReceiveWithdrawExpired,
    ReceiveWithdrawRequest,
)
from raiden.ui.startup import RaidenBundle, ServicesBundle
from raiden.utils.formatting import lpex, to_checksum_address
from raiden.utils.gevent import spawn_named
from raiden.utils.logging import redact_secret
from raiden.utils.runnable import Runnable
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.signer import LocalSigner, Signer
from raiden.utils.transfers import random_secret
from raiden.utils.typing import (
    MYPY_ANNOTATION,
    Address,
    AddressMetadata,
    BlockNumber,
    BlockTimeout,
    InitiatorAddress,
    MonitoringServiceAddress,
    OneToNAddress,
    PaymentAmount,
    PaymentID,
    PrivateKey,
    Secret,
    SecretHash,
    SecretRegistryAddress,
    TargetAddress,
    TokenNetworkAddress,
    WithdrawAmount,
    typecheck,
)
from raiden.utils.upgrades import UpgradeManager
from raiden_contracts.constants import ChannelEvent
from raiden_contracts.contract_manager import ContractManager

log = structlog.get_logger(__name__)
StatusesDict = Dict[TargetAddress, Dict[PaymentID, "PaymentStatus"]]

PFS_UPDATE_CAPACITY_STATE_CHANGES = (
    ContractReceiveChannelDeposit,
    ReceiveUnlock,
    ReceiveWithdrawRequest,
    ReceiveWithdrawExpired,
    ReceiveTransferCancelRoute,
    ReceiveLockExpired,
    ReceiveTransferRefund,
    # State change | Reason why update is not needed
    # ActionInitInitiator | Update triggered by SendLockedTransfer
    # ActionInitMediator | Update triggered by SendLockedTransfer
    # ActionInitTarget | Update triggered by SendLockedTransfer
    # ActionTransferReroute | Update triggered by SendLockedTransfer
    # ActionChannelWithdraw | Upd. triggered by ReceiveWithdrawConfirmation/ReceiveWithdrawExpired
)
PFS_UPDATE_CAPACITY_EVENTS = (
    SendUnlock,
    SendLockedTransfer,
    SendWithdrawRequest,
    SendWithdrawExpired,
)

# Assume lower capacity for fees when in doubt, see
# https://raiden-network-specification.readthedocs.io/en/latest/pathfinding_service.html
#    #when-to-send-pfsfeeupdates
PFS_UPDATE_FEE_STATE_CHANGES = (
    ContractReceiveChannelDeposit,
    ReceiveWithdrawRequest,
    ReceiveWithdrawExpired,
)
PFS_UPDATE_FEE_EVENTS = (SendWithdrawRequest, SendWithdrawExpired)

assert not set(PFS_UPDATE_FEE_STATE_CHANGES) - set(
    PFS_UPDATE_CAPACITY_STATE_CHANGES
), "No fee updates without capacity updates possible"
assert not set(PFS_UPDATE_FEE_EVENTS) - set(
    PFS_UPDATE_CAPACITY_EVENTS
), "No fee updates without capacity updates possible"


def initiator_init(
    raiden: "RaidenService",
    transfer_identifier: PaymentID,
    transfer_amount: PaymentAmount,
    transfer_secret: Secret,
    transfer_secrethash: SecretHash,
    token_network_address: TokenNetworkAddress,
    target_address: TargetAddress,
    lock_timeout: Optional[BlockTimeout] = None,
    route_states: Optional[List[RouteState]] = None,
) -> Tuple[Optional[str], ActionInitInitiator]:
    transfer_state = TransferDescriptionWithSecretState(
        token_network_registry_address=raiden.default_registry.address,
        payment_identifier=transfer_identifier,
        amount=transfer_amount,
        token_network_address=token_network_address,
        initiator=InitiatorAddress(raiden.address),
        target=target_address,
        secret=transfer_secret,
        secrethash=transfer_secrethash,
        lock_timeout=lock_timeout,
    )

    error_msg = None
    if route_states is None:
        our_address_metadata = raiden.transport.address_metadata

        msg = "Transport is not initialized with raiden-service"
        assert our_address_metadata is not None, msg

        error_msg, route_states, feedback_token = routing.get_best_routes(
            chain_state=views.state_from_raiden(raiden),
            token_network_address=token_network_address,
            one_to_n_address=raiden.default_one_to_n_address,
            from_address=InitiatorAddress(raiden.address),
            to_address=target_address,
            amount=transfer_amount,
            previous_address=None,
            privkey=raiden.privkey,
            our_address_metadata=our_address_metadata,
            pfs_proxy=raiden.pfs_proxy,
        )

        # Only prepare feedback when token is available
        if feedback_token is not None:
            for route_state in route_states:
                raiden.route_to_feedback_token[tuple(route_state.route)] = feedback_token

    return error_msg, ActionInitInitiator(transfer_state, route_states)


def smart_contract_filters_from_node_state(
    chain_state: ChainState,
    secret_registry_address: SecretRegistryAddress,
    service_registry: Optional[ServiceRegistry],
) -> RaidenContractFilter:
    token_network_registries = chain_state.identifiers_to_tokennetworkregistries.values()
    token_networks = [tn for tnr in token_network_registries for tn in tnr.token_network_list]
    channels_of_token_network = {
        tn.address: set(tn.channelidentifiers_to_channels.keys())
        for tn in token_networks
        if tn.channelidentifiers_to_channels
    }

    return RaidenContractFilter(
        secret_registry_address=secret_registry_address,
        token_network_registry_addresses={tnr.address for tnr in token_network_registries},
        token_network_addresses={tn.address for tn in token_networks},
        channels_of_token_network=channels_of_token_network,
        ignore_secret_registry_until_channel_found=not channels_of_token_network,
        service_registry=service_registry,
    )


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

    def matches(self, token_network_address: TokenNetworkAddress, amount: PaymentAmount) -> bool:
        return token_network_address == self.token_network_address and amount == self.amount


class SyncTimeout:
    """Helper to determine if the sync should halt or continue.

    The goal of this helper is to stop syncing before the block
    `current_confirmed_head` is pruned, otherwise JSON-RPC requests will start
    to fail.
    """

    def __init__(self, current_confirmed_head: BlockNumber, timeout: float) -> None:
        self.sync_start = time.monotonic()
        self.timeout = timeout
        self.current_confirmed_head = current_confirmed_head

    def time_elapsed(self) -> float:
        delta = time.monotonic() - self.sync_start
        return delta

    def should_continue(self, last_fetched_block: BlockNumber) -> bool:
        has_time = self.timeout >= self.time_elapsed()
        has_blocks_unsynched = self.current_confirmed_head > last_fetched_block

        return has_time and has_blocks_unsynched


class SynchronizationState(Enum):
    FULLY_SYNCED = "fully_synced"
    PARTIALLY_SYNCED = "partially_synced"


class RaidenService(Runnable):
    """A Raiden node."""

    def __init__(
        self,
        rpc_client: JSONRPCClient,
        proxy_manager: ProxyManager,
        query_start_block: BlockNumber,
        raiden_bundle: RaidenBundle,
        services_bundle: Optional[ServicesBundle],
        transport: MatrixTransport,
        raiden_event_handler: EventHandler,
        message_handler: MessageHandler,
        routing_mode: RoutingMode,
        config: RaidenConfig,
        api_server: Optional[APIServer] = None,
        pfs_proxy: Optional[PFSProxy] = None,
    ) -> None:
        super().__init__()

        # check that the settlement timeout fits the limits of the contract
        settlement_timeout_min = raiden_bundle.token_network_registry.settlement_timeout_min(
            BLOCK_ID_LATEST
        )
        settlement_timeout_max = raiden_bundle.token_network_registry.settlement_timeout_max(
            BLOCK_ID_LATEST
        )
        invalid_settle_timeout = (
            config.settle_timeout < settlement_timeout_min
            or config.settle_timeout > settlement_timeout_max
            or config.settle_timeout < config.reveal_timeout * 2
        )
        if invalid_settle_timeout:
            contract = to_checksum_address(raiden_bundle.token_network_registry.address)
            raise InvalidSettleTimeout(
                (
                    f"Settlement timeout for Registry contract {contract} must "
                    f"be in range [{settlement_timeout_min}, {settlement_timeout_max}], "
                    f"is {config.settle_timeout}"
                )
            )

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
        self.notifications: Dict = {}  # notifications are unique (and indexed) by id.

        self.signer: Signer = LocalSigner(self.rpc_client.privkey)
        self.address: Address = self.signer.address
        self.transport: MatrixTransport = transport

        self.alarm = AlarmTask(
            proxy_manager=proxy_manager, sleep_time=self.config.blockchain.query_interval
        )
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
        self.stop_event.set()  # inits as stopped
        self.greenlets: List[Greenlet] = []

        self.last_log_time = time.monotonic()
        self.last_log_block = BlockNumber(0)

        self.contract_manager: ContractManager = ContractManager(config.contracts_path)
        self.wal: Optional[WriteAheadLog] = None
        self.db_lock: Optional[filelock.UnixFileLock] = None

        if pfs_proxy is None:
            assert config.pfs_config is not None, "must not be None"
            pfs_proxy = PFSProxy(config.pfs_config)
        self.pfs_proxy = pfs_proxy

        if self.config.database_path != ":memory:":
            database_dir = os.path.dirname(config.database_path)
            os.makedirs(database_dir, exist_ok=True)

            self.database_dir: Optional[str] = database_dir

            # Two raiden processes must not write to the same database. Even
            # though it's possible the database itself would not be corrupt,
            # the node's state could. If a database was shared among multiple
            # nodes, the database WAL would be the union of multiple node's
            # WAL. During a restart a single node can't distinguish its state
            # changes from the others, and it would apply it all, meaning that
            # a node would execute the actions of itself and the others.
            #
            # Additionally the database snapshots would be corrupt, because it
            # would not represent the effects of applying all the state changes
            # in order.
            lock_file = os.path.join(self.database_dir, ".lock")
            self.db_lock = filelock.FileLock(lock_file)
        else:
            self.database_dir = None
            self.serialization_file = None
            self.db_lock = None

        self.payment_identifier_lock = gevent.lock.Semaphore()

        # A list is not hashable, so use tuple as key here
        self.route_to_feedback_token: Dict[Tuple[Address, ...], UUID] = {}

        # Flag used to skip the processing of all Raiden events during the
        # startup.
        #
        # Rationale: At the startup, the latest snapshot is restored and all
        # state changes which are not 'part' of it are applied. The criteria to
        # re-apply the state changes is their 'absence' in the snapshot, /not/
        # their completeness. Because these state changes are re-executed
        # in-order and some of their side-effects will already have been
        # completed, the events should be delayed until the state is
        # synchronized (e.g. an open channel state change, which has already
        # been mined).
        #
        # Incomplete events, i.e. the ones which don't have their side-effects
        # applied, will be executed once the blockchain state is synchronized
        # because of the node's queues.
        self.ready_to_process_events = False

        # Counters used for state snapshotting
        self.state_change_qty_snapshot = 0
        self.state_change_qty = 0

    def start(self) -> None:
        """Start the node synchronously. Raises directly if anything went wrong on startup"""
        assert self.stop_event.ready(), f"Node already started. node:{self!r}"
        self.stop_event.clear()
        self.greenlets = []

        self.ready_to_process_events = False  # set to False because of restarts

        self._initialize_wal()
        self._synchronize_with_blockchain()

        chain_state = views.state_from_raiden(self)

        self._initialize_payment_statuses(chain_state)
        self._initialize_transactions_queues(chain_state)
        self._initialize_messages_queues(chain_state)
        self._initialize_channel_fees()
        self._initialize_monitoring_services_queue(chain_state)
        self._initialize_ready_to_process_events()

        # Start the side-effects:
        # - React to blockchain events
        # - React to incoming messages
        # - Send pending transactions
        # - Send pending message
        self.alarm.greenlet.link_exception(self.on_error)
        self.transport.greenlet.link_exception(self.on_error)
        if self.api_server is not None:
            self.api_server.greenlet.link_exception(self.on_error)
        self._start_transport()
        self._start_alarm_task()

        log.debug("Raiden Service started", node=to_checksum_address(self.address))
        super().start()

        self._set_rest_api_service_available()

    def _run(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=method-hidden
        """Busy-wait on long-lived subtasks/greenlets, re-raise if any error occurs"""
        self.greenlet.name = f"RaidenService._run node:{to_checksum_address(self.address)}"
        try:
            self.stop_event.wait()
        except gevent.GreenletExit:  # killed without exception
            self.stop_event.set()
            gevent.killall([self.alarm.greenlet, self.transport.greenlet])  # kill children
            raise  # re-raise to keep killed status

    def stop(self) -> None:
        """Stop the node gracefully. Raise if any stop-time error occurred on any subtask"""
        if self.stop_event.ready():  # not started
            return

        # Needs to come before any greenlets joining
        self.stop_event.set()

        # Filters must be uninstalled after the alarm task has stopped. Since
        # the events are polled by an alarm task callback, if the filters are
        # uninstalled before the alarm task is fully stopped the callback will
        # fail.
        #
        # We need a timeout to prevent an endless loop from trying to
        # contact the disconnected client
        if self.api_server is not None:
            self.api_server.stop()
            self.api_server.greenlet.join()

        self.alarm.stop()
        self.alarm.greenlet.join()

        self.transport.stop()
        self.transport.greenlet.join()

        assert (
            self.blockchain_events
        ), f"The blockchain_events has to be set by the start. node:{self!r}"
        self.blockchain_events.stop()

        # Close storage DB to release internal DB lock
        assert (
            self.wal
        ), f"The Service must have been started before it can be stopped. node:{self!r}"
        self.wal.storage.close()
        self.wal = None

        if self.db_lock is not None:
            self.db_lock.release()

        log.debug("Raiden Service stopped", node=to_checksum_address(self.address))

    def add_notification(
        self,
        notification: Notification,
        log_opts: Optional[Dict] = None,
        click_opts: Optional[Dict] = None,
    ) -> None:
        log_opts = log_opts or {}
        click_opts = click_opts or {}

        log.info(notification.summary, **log_opts)
        click.secho(notification.body, **click_opts)

        self.notifications[notification.id] = notification

    @property
    def confirmation_blocks(self) -> BlockTimeout:
        return self.config.blockchain.confirmation_blocks

    @property
    def privkey(self) -> PrivateKey:
        return self.rpc_client.privkey

    def add_pending_greenlet(self, greenlet: Greenlet) -> None:
        """Ensures an error on the passed greenlet crashes self/main greenlet."""

        def remove(_: Any) -> None:
            self.greenlets.remove(greenlet)

        self.greenlets.append(greenlet)
        greenlet.link_exception(self.on_error)
        greenlet.link_value(remove)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} node:{to_checksum_address(self.address)}>"

    def _start_transport(self) -> None:
        """Initialize the transport and related facilities.

        Note:
            The node has first to `_synchronize_with_blockchain` before
            starting the transport. This synchronization includes the on-chain
            channel state and is necessary to reject new messages for closed
            channels.
        """
        assert self.ready_to_process_events, f"Event processing disabled. node:{self!r}"
        msg = (
            "`self.blockchain_events` is `None`. "
            "Seems like `_synchronize_with_blockchain` wasn't called before `_start_transport`."
        )
        assert self.blockchain_events is not None, msg

        if self.default_service_registry is not None:
            populate_services_addresses(
                self.transport, self.default_service_registry, BLOCK_ID_LATEST
            )
        self.transport.start(raiden_service=self, prev_auth_data=None)

    def _make_initial_state(self) -> ChainState:
        # On first run Raiden needs to fetch all events for the payment
        # network, to reconstruct all token network graphs and find opened
        # channels
        #
        # The value `self.query_start_block` is an optimization, because
        # Raiden has to poll all events until the last confirmed block,
        # using the genesis block would result in fetchs for a few million
        # of unnecessary blocks. Instead of querying all these unnecessary
        # blocks, the configuration variable `query_start_block` is used to
        # start at the block which `TokenNetworkRegistry`  was deployed.
        last_log_block_number = self.query_start_block
        last_log_block_hash = self.rpc_client.blockhash_from_blocknumber(last_log_block_number)

        initial_state = ChainState(
            pseudo_random_generator=random.Random(),
            block_number=last_log_block_number,
            block_hash=last_log_block_hash,
            our_address=self.address,
            chain_id=self.rpc_client.chain_id,
        )
        token_network_registry_address = self.default_registry.address
        token_network_registry = TokenNetworkRegistryState(
            token_network_registry_address,
            [],  # empty list of token network states as it's the node's startup
        )
        initial_state.identifiers_to_tokennetworkregistries[
            token_network_registry_address
        ] = token_network_registry

        return initial_state

    def _initialize_wal(self) -> None:
        if self.database_dir is not None:
            try:
                assert (
                    self.db_lock is not None
                ), "If a database_dir is present, a lock for the database has to exist"
                self.db_lock.acquire(timeout=0)
                assert self.db_lock.is_locked, f"Database not locked. node:{self!r}"
            except (filelock.Timeout, AssertionError) as ex:
                raise RaidenUnrecoverableError(
                    "Could not aquire database lock. Maybe a Raiden node for this account "
                    f"({to_checksum_address(self.address)}) is already running?"
                ) from ex

        self.maybe_upgrade_db()

        storage = sqlite.SerializedSQLiteStorage(
            database_path=self.config.database_path, serializer=JSONSerializer()
        )
        storage.update_version()
        storage.log_run()

        try:
            initial_state = self._make_initial_state()
            (
                state_snapshot,
                state_change_start,
                state_change_qty_snapshot,
            ) = wal.restore_or_init_snapshot(
                storage=storage, node_address=self.address, initial_state=initial_state
            )

            state, state_change_qty_unapplied = wal.replay_state_changes(
                node_address=self.address,
                state=state_snapshot,
                state_change_range=Range(state_change_start, HIGH_STATECHANGE_ULID),
                storage=storage,
                transition_function=node.state_transition,  # type: ignore
            )
        except SerializationError:
            raise RaidenUnrecoverableError(
                "Could not restore state. "
                "It seems like the existing database is incompatible with "
                "the current version of Raiden. Consider using a stable "
                "version of the Raiden client."
            )

        if state_change_qty_snapshot == 0:
            print(
                "This is the first time Raiden is being used with this address. "
                "Processing all the events may take some time. Please wait ..."
            )

        self.state_change_qty_snapshot = state_change_qty_snapshot
        self.state_change_qty = state_change_qty_snapshot + state_change_qty_unapplied

        msg = "The state must be a ChainState instance."
        assert isinstance(state, ChainState), msg

        self.wal = WriteAheadLog(state, storage, node.state_transition)

        # The `Block` state change is dispatched only after all the events
        # for that given block have been processed, filters can be safely
        # installed starting from this position without losing events.
        last_log_block_number = views.block_number(self.wal.get_current_state())
        log.debug(
            "Querying blockchain from block",
            last_restored_block=last_log_block_number,
            node=to_checksum_address(self.address),
        )

        known_networks = views.get_token_network_registry_address(views.state_from_raiden(self))
        if known_networks and self.default_registry.address not in known_networks:
            configured_registry = to_checksum_address(self.default_registry.address)
            known_registries = lpex(known_networks)
            raise RuntimeError(
                f"Token network address mismatch.\n"
                f"Raiden is configured to use the smart contract "
                f"{configured_registry}, which conflicts with the current known "
                f"smart contracts {known_registries}"
            )

    def _log_sync_progress(
        self, polled_block_number: BlockNumber, target_block: BlockNumber
    ) -> None:
        """Print a message if there are many blocks to be fetched, or if the
        time in-between polls is high.
        """
        now = time.monotonic()
        blocks_until_target = target_block - polled_block_number
        polled_block_count = polled_block_number - self.last_log_block
        elapsed = now - self.last_log_time

        if blocks_until_target > 100 or elapsed > 15.0:
            log.info(
                "Synchronizing blockchain events",
                remaining_blocks_to_sync=blocks_until_target,
                blocks_per_second=polled_block_count / elapsed,
                to_block=target_block,
                elapsed=elapsed,
            )
            self.last_log_time = time.monotonic()
            self.last_log_block = polled_block_number

    def _synchronize_with_blockchain(self) -> None:
        """Prepares the alarm task callback and synchronize with the blockchain
        since the last run.

         Notes about setup order:
         - The filters must be polled after the node state has been primed,
           otherwise the state changes won't have effect.
         - The synchronization must be done before the transport is started, to
           reject messages for closed/settled channels.
        """
        msg = (
            f"Transport must not be started before the node has synchronized "
            f"with the blockchain, otherwise the node may accept transfers to a "
            f"closed channel. node:{self!r}"
        )
        assert not self.transport, msg
        assert self.wal, f"The database must have been initialized. node:{self!r}"

        chain_state = views.state_from_raiden(self)

        # The `Block` state change is dispatched only after all the events for
        # that given block have been processed, filters can be safely installed
        # starting from this position without missing events.
        last_block_number = views.block_number(chain_state)

        event_filter = smart_contract_filters_from_node_state(
            chain_state,
            self.default_secret_registry.address,
            self.default_service_registry,
        )

        log.debug("initial filter", event_filter=event_filter, node=self.address)
        blockchain_events = BlockchainEvents(
            web3=self.rpc_client.web3,
            chain_id=chain_state.chain_id,
            contract_manager=self.contract_manager,
            last_fetched_block=last_block_number,
            event_filter=event_filter,
            block_batch_size_config=self.config.blockchain.block_batch_size_config,
            node_address=self.address,
        )
        blockchain_events.register_listener(self._blockchain_event_listener)

        self.last_log_block = last_block_number
        self.last_log_time = time.monotonic()

        # `blockchain_events` is a requirement for
        # `_best_effort_synchronize_with_confirmed_head`, so it must be set
        # before calling it
        self.blockchain_events = blockchain_events

        synchronization_state = SynchronizationState.PARTIALLY_SYNCED
        while synchronization_state is SynchronizationState.PARTIALLY_SYNCED:
            latest_block = self.rpc_client.get_block(block_identifier=BLOCK_ID_LATEST)
            synchronization_state = self._best_effort_synchronize(latest_block)

        self.alarm.register_callback(self._best_effort_synchronize)

    def _blockchain_event_listener(self, events: List[DecodedEvent]) -> None:
        if not self.transport.started:
            return
        for event in events:
            args = event.event_data["args"]
            if event.event_data["event"] == ChannelEvent.OPENED:
                other = (
                    args["participant1"]
                    if args["participant1"] != self.address
                    else args["participant2"]
                )
                self.transport.health_check_web_rtc(other)

    def _start_alarm_task(self) -> None:
        """Start the alarm task.

        Note:
            The alarm task must be started only when processing events is
            allowed, otherwise side-effects of blockchain events will be
            ignored.
        """
        assert self.ready_to_process_events, f"Event processing disabled. node:{self!r}"
        self.alarm.start()

    def _set_rest_api_service_available(self) -> None:
        if self.raiden_api:
            assert self.rest_api, "api enabled in config but self.rest_api not initialized"
            self.rest_api.raiden_api = self.raiden_api
            print("Synchronization complete, REST API services now available.")

    def _initialize_ready_to_process_events(self) -> None:
        """Mark the node as ready to start processing raiden events that may
        send messages or transactions.

        This flag /must/ be set to true before the both  transport and the
        alarm are started.
        """
        msg = (
            f"The transport must not be initialized before the "
            f"`ready_to_process_events` flag is set, since this is a requirement "
            f"for the alarm task and the alarm task should be started before the "
            f"transport to avoid race conditions. node:{self!r}"
        )
        assert not self.transport, msg
        msg = (
            f"Alarm task must not be started before the "
            f"`ready_to_process_events` flag is set, otherwise events may be "
            f"missed. node:{self!r}"
        )
        assert not self.alarm, msg

        self.ready_to_process_events = True

    def get_block_number(self) -> BlockNumber:
        assert self.wal, f"WAL object not yet initialized. node:{self!r}"
        return views.block_number(self.wal.get_current_state())

    def on_messages(self, messages: List[Message]) -> None:
        self.message_handler.on_messages(self, messages)

    def handle_and_track_state_changes(self, state_changes: List[StateChange]) -> None:
        """Dispatch the state change and does not handle the exceptions.

        When the method is used the exceptions are tracked and re-raised in the
        raiden service thread.
        """
        if len(state_changes) == 0:
            return

        # It's important to /not/ block here, because this function can
        # be called from the alarm task greenlet, which should not
        # starve. This was a problem when the node decided to send a new
        # transaction, since the proxies block until the transaction is
        # mined and confirmed (e.g. the settle window is over and the
        # node sends the settle transaction).
        for greenlet in self.handle_state_changes(state_changes):
            self.add_pending_g