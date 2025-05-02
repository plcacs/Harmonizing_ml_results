""" Utilities to make and assert transfers. """
import functools
import itertools
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Dict, Generator, Iterable, Iterator, Optional, Sequence, Set, Tuple, TypeVar, Union, cast

import gevent
from gevent.timeout import Timeout
from raiden.constants import EMPTY_SIGNATURE
from raiden.message_handler import MessageHandler
from raiden.messages.abstract import SignedMessage
from raiden.messages.decode import balanceproof_from_envelope
from raiden.messages.metadata import Metadata, RouteMetadata
from raiden.messages.transfers import Lock, LockedTransfer, LockExpired, Unlock
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_MEDIATION_FEE_MARGIN, DEFAULT_RETRY_TIMEOUT, INTERNAL_ROUTING_DEFAULT_FEE_PERC
from raiden.storage.restore import get_event_with_balance_proof_by_balance_hash, get_state_change_with_balance_proof_by_locksroot, get_state_change_with_transfer_by_secrethash
from raiden.storage.wal import SavedState, WriteAheadLog
from raiden.tests.utils.events import has_unlock_failure, raiden_state_changes_search_for_item
from raiden.tests.utils.factories import create_route_states_from_routes, make_initiator_address, make_message_identifier, make_secret_with_hash, make_target_address
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.transfer import channel, views
from raiden.transfer.architecture import TransitionResult
from raiden.transfer.channel import compute_locksroot
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed, EventUnlockFailed, SendSecretRequest
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState
from raiden.transfer.mediated_transfer.state_change import ActionInitMediator, ActionInitTarget, ReceiveLockExpired, ReceiveTransferRefund
from raiden.transfer.state import BalanceProofSignedState, BalanceProofUnsignedState, ChannelState, HashTimeLockState, NettingChannelState, PendingLocksState, RouteState, make_empty_pending_locks_state
from raiden.transfer.state_change import ContractReceiveChannelDeposit, ReceiveUnlock
from raiden.utils.formatting import to_checksum_address
from raiden.utils.signer import LocalSigner, Signer
from raiden.utils.timeout import BlockTimeout
from raiden.utils.typing import (
    MYPY_ANNOTATION, Address, Any, Balance, BlockNumber, BlockTimeout as BlockOffset,
    Callable, ChainID, FeeAmount, List, LockedAmount, Nonce, Optional, PaymentAmount,
    PaymentID, PaymentWithFeeAmount, SecretHash, TargetAddress, TokenAddress,
    TokenAmount, TokenNetworkAddress, cast, typecheck
)

ZERO_FEE = FeeAmount(0)

class TransferState(Enum):
    """Represents the target state of a transfer."""
    LOCKED = 'locked'
    UNLOCKED = 'unlocked'
    EXPIRED = 'expired'
    SECRET_NOT_REVEALED = 'secret_not_revealed'
    SECRET_NOT_REQUESTED = 'secret_not_requested'
    SECRET_REVEALED = 'secret_revealed'

def sign_and_inject(message: SignedMessage, signer: Signer, app: RaidenService) -> None:
    """Sign the message with key and inject it directly in the app transport layer."""
    message.sign(signer)
    MessageHandler().on_messages(app, [message])

def get_channelstate(app0: RaidenService, app1: RaidenService, token_network_address: TokenNetworkAddress) -> NettingChannelState:
    channel_state = views.get_channelstate_by_token_network_and_partner(
        views.state_from_raiden(app0),
        token_network_address,
        app1.address
    )
    assert channel_state
    return channel_state

def create_route_state_for_route(
    apps: Sequence[RaidenService],
    token_address: TokenAddress,
    fee_estimate: Optional[FeeAmount] = None
) -> RouteState:
    assert len(apps) > 1, 'Need at least two nodes for a route'
    route: List[Address] = []
    address_metadata: Dict[Address, Dict[str, Any]] = {}
    for app in apps:
        route.append(app.address)
        address_metadata[app.address] = app.transport.address_metadata
    token_network = views.get_token_network_by_token_address(
        views.state_from_raiden(apps[0]),
        apps[0].default_registry.address,
        token_address
    )
    assert token_network
    if fee_estimate is not None:
        return RouteState(
            route=route,
            address_to_metadata=address_metadata,
            estimated_fee=fee_estimate
        )
    else:
        return RouteState(
            route=route,
            address_to_metadata=address_metadata
        )

@contextmanager
def patch_transfer_routes(
    routes: Sequence[Sequence[RaidenService]],
    token_address: TokenAddress
) -> Generator[None, None, None]:
    """
    Context manager to set specific routes for transfers.
    This circumvents the lack of a PFS in the tests making a transfer fail.
    """
    apps = set(itertools.chain.from_iterable(routes))
    for app in apps:
        app.__mediated_transfer_async = app.mediated_transfer_async
        route_states = [create_route_state_for_route(route, token_address) for route in routes]
        app.mediated_transfer_async = functools.partial(
            app.__mediated_transfer_async,
            route_states=route_states
        )
    yield
    for app in apps:
        app.mediated_transfer_async = app.__mediated_transfer_async
        del app.__mediated_transfer_async

@contextmanager
def watch_for_unlock_failures(*apps: RaidenService) -> Generator[None, None, None]:
    """
    Context manager to assure there are no failing unlocks during transfers in integration tests.
    """
    failed_event: Optional[Union[EventUnlockClaimFailed, EventUnlockFailed]] = None

    def check(event: Any) -> None:
        nonlocal failed_event
        if isinstance(event, (EventUnlockClaimFailed, EventUnlockFailed)):
            failed_event = event
    for app in apps:
        app.raiden_event_handler.pre_hooks.add(check)
    try:
        yield
    finally:
        for app in apps:
            app.raiden_event_handler.pre_hooks.remove(check)
        assert failed_event is None, f'Unexpected unlock failure: {str(failed_event)}'

def transfer(
    initiator_app: RaidenService,
    target_app: RaidenService,
    token_address: TokenAddress,
    amount: PaymentAmount,
    identifier: PaymentID,
    timeout: Optional[int] = None,
    transfer_state: TransferState = TransferState.UNLOCKED,
    expect_unlock_failures: bool = False,
    routes: Optional[Sequence[Sequence[RaidenService]]] = None
) -> SecretHash:
    """Nice to read shortcut to make successful mediated transfer.

    Note:
        Only the initiator and target are synced.
    """
    route_states = None
    if routes:
        route_states = []
        for route in routes:
            route_states.append(create_route_state_for_route(route, token_address))
    if transfer_state is TransferState.UNLOCKED:
        return _transfer_unlocked(
            initiator_app=initiator_app,
            target_app=target_app,
            token_address=token_address,
            amount=amount,
            identifier=identifier,
            timeout=timeout,
            expect_unlock_failures=expect_unlock_failures,
            route_states=route_states
        )
    elif transfer_state is TransferState.EXPIRED:
        return _transfer_expired(
            initiator_app=initiator_app,
            target_app=target_app,
            token_address=token_address,
            amount=amount,
            identifier=identifier,
            timeout=timeout
        )
    elif transfer_state is TransferState.SECRET_NOT_REQUESTED:
        return _transfer_secret_not_requested(
            initiator_app=initiator_app,
            target_app=target_app,
            token_address=token_address,
            amount=amount,
            identifier=identifier,
            timeout=timeout,
            route_states=route_states
        )
    elif transfer_state is TransferState.LOCKED:
        return _transfer_locked(
            initiator_app=initiator_app,
            target_app=target_app,
            token_address=token_address,
            amount=amount,
            identifier=identifier,
            timeout=timeout,
            route_states=route_states
        )
    else:
        raise RuntimeError('Type of transfer not implemented.')

def _transfer_unlocked(
    initiator_app: RaidenService,
    target_app: RaidenService,
    token_address: TokenAddress,
    amount: PaymentAmount,
    identifier: PaymentID,
    timeout: Optional[int] = None,
    expect_unlock_failures: bool = False,
    route_states: Optional[Sequence[RouteState]] = None
) -> SecretHash:
    assert isinstance(target_app.message_handler, WaitForMessage)
    if timeout is None:
        timeout = 10
    wait_for_unlock = target_app.message_handler.wait_for_message(
        Unlock,
        {'payment_identifier': identifier}
    )
    token_network_registry_address = initiator_app.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(
        chain_state=views.state_from_raiden(initiator_app),
        token_network_registry_address=token_network_registry_address,
        token_address=token_address
    )
    assert token_network_address
    secret, secrethash = make_secret_with_hash()
    payment_status = initiator_app.mediated_transfer_async(
        token_network_address=token_network_address,
        amount=amount,
        target=TargetAddress(target_app.address),
        identifier=identifier,
        secret=secret,
        secrethash=secrethash,
        route_states=route_states
    )
    apps = [initiator_app, target_app]
    with watch_for_unlock_failures(*apps) if not expect_unlock_failures else nullcontext():
        with Timeout(seconds=timeout):
            wait_for_unlock.get()
            msg = f'transfer from {to_checksum_address(initiator_app.address)} to {to_checksum_address(target_app.address)} failed.'
            assert payment_status.payment_done.get(), msg
    return secrethash

def _transfer_expired(
    initiator_app: RaidenService,
    target_app: RaidenService,
    token_address: TokenAddress,
    amount: PaymentAmount,
    identifier: PaymentID,
    timeout: Optional[int] = None
) -> SecretHash:
    assert identifier is not None, 'The identifier must be provided'
    assert isinstance(target_app.message_handler, WaitForMessage)
    if timeout is None:
        timeout = 90
    secret, secrethash = make_secret_with_hash()
    wait_for_remove_expired_lock = target_app.message_handler.wait_for_message(
        LockExpired,
        {'secrethash': secrethash}
    )
    token_network_registry_address = initiator_app.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(
        chain_state=views.state_from_raiden(initiator_app),
        token_network_registry_address=token_network_registry_address,
        token_address=token_address
    )
    assert token_network_address
    payment_status = initiator_app.mediated_transfer_async(
        token_network_address=token_network_address,
        amount=amount,
        target=TargetAddress(target_app.address),
        identifier=identifier,
        secret=secret,
        secrethash=secrethash
    )
    with Timeout(seconds=timeout):
        wait_for_remove_expired_lock.get()
        msg = f'transfer from {to_checksum_address(initiator_app.address)} to {to_checksum_address(target_app.address)} did not expire.'
        assert payment_status.payment_done.get() is False, msg
    return secrethash

def _transfer_secret_not_requested(
    initiator_app: RaidenService,
    target_app: RaidenService,
    token_address: TokenAddress,
    amount: PaymentAmount,
    identifier: PaymentID,
    timeout: Optional[int] = None,
    route_states: Optional[Sequence[RouteState]] = None
) -> SecretHash:
    if timeout is None:
        timeout = 10
    secret, secrethash = make_secret_with_hash()
    assert isinstance(target_app.raiden_event_handler, HoldRaidenEventHandler)
    hold_secret_request = target_app.raiden_event_handler.hold(
        SendSecretRequest,
        {'secrethash': secrethash}
    )
    token_network_registry_address = initiator_app.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(
        chain_state=views.state_from_raiden(initiator_app),
        token_network_registry_address=token_network_registry_address,
        token_address=token_address
    )
    assert token_network_address
    initiator_app.mediated_transfer_async(
        token_network_address=token_network_address,
        amount=amount,
        target=TargetAddress(target_app.address),
        identifier=identifier,
        secret=secret,
        secrethash=secrethash,
        route_states=route_states
    )
    with Timeout(seconds=timeout):
        hold_secret_request.get()
    return secrethash

def _transfer_locked(
    initiator_app: RaidenService,
    target_app: RaidenService,
    token_address: TokenAddress,
    amount: PaymentAmount,
    identifier: PaymentID,
    timeout: Optional[int] = None,
    route_states: Optional[Sequence[RouteState]] = None
) -> SecretHash:
    if timeout is None:
        timeout = 10
    secret, secrethash = make_secret_with_hash()
    token_network_registry_address = initiator_app.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(
        chain_state=views.state_from_raiden(initiator_app),
        token_network_registry_address=token_network_registry_address,
        token_address=token_address
    )
    assert token_network_address is not None
    initiator_app.mediated_transfer_async(
        token_network_address=token_network_address,
        amount=amount,
        target=TargetAddress(target_app.address),
        identifier=identifier,
        secret=secret,
        secrethash=secrethash,
        route_states=route_states
    )
    return secrethash

def transfer_and_assert_path(
    path: Sequence[RaidenService],
    token_address: TokenAddress,
    amount: PaymentAmount,
    identifier: PaymentID,
    timeout: int = 10,
    fee_estimate: FeeAmount = FeeAmount(0)
) -> SecretHash:
    """Nice to read shortcut to make successful LockedTransfer.

    Note:
        This utility *does not enforce the path*, however it does check the
        provided path is used in totality. It's the responsability of the
        caller to ensure the path will be used. All nodes in `path` are
        synched.
    """
    assert identifier is not None, 'The identifier must be provided'
    secret, secrethash = make_secret_with_hash()
    first_app = path[0]
    token_network_registry_address = first_app.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(
        chain_state=views.state_from_raiden(first_app),
        token_network_registry_address=token_network_registry_address,
        token_address=token_address
    )
    assert token_network_address
    for app in path:
        assert isinstance(app.message_handler, WaitForMessage)
        msg = 'The apps must be on the same token network registry'
        assert app.default_registry.address == token_network_registry_address, msg
        app_token_network_address = views.get_token_network_address_by_token_address(
            chain_state=views.state_from_raiden(app),
            token_network_registry_address=token_network_registry_address,
            token_address=token_address
        )
        msg = 'The apps must be synchronized with the blockchain'
        assert token_network_address == app_token_network_address, msg
    pairs = zip(path[:-1], path[1:])
    receiving: List[Tuple[RaidenService, int]] = []
    for from_app, to_app in pairs:
        from_channel_state = views.get_channelstate_by_token_network_and_partner(
            chain_state=views.state_from_raiden(from_app),
            token_network_address=token_network_address,
            partner_address=to_app.address
        )
        to_channel_state = views.get_channelstate_by_token_network_and_partner(
            chain_state=views.state_from_raiden(to_app),
            token_network_address=token_network_address,
            partner_address=from_app.address
        )
        msg = f'{to_checksum_address(from_app.address)} does not have a channel with {to_checksum_address(to_app.address)} needed to transfer through the path {[to_checksum_address(app.address) for app in path]}.'
        assert from_channel_state, msg
        assert to_channel_state, msg
        msg = f'channel among {to_checksum_address(from_app.address)} and {to_checksum_address(to_app.address)} must be open to be used for a transfer'
        assert channel.get_status(from_channel_state) == ChannelState.STATE_OPENED, msg
        assert channel.get_status(to_channel_state) == ChannelState.STATE_OPENED, msg
        receiving.append((to_app, to_channel_state.identifier))
    assert isinstance(app.message_handler, WaitForMessage)
    results = set((app.message_handler.wait_for_message(
        Unlock,
        {
            'channel_identifier': channel_identifier,
            'token_network_address': token_network_address,
            'payment_identifier': identifier,
            'secret': secret
        }
    ) for app, channel_identifier in receiving))
    last_app = path[-1]
    payment_status = first_app.mediated_transfer_async(
        token_network_address=token_network_address,
        amount=amount,
        target=TargetAddress