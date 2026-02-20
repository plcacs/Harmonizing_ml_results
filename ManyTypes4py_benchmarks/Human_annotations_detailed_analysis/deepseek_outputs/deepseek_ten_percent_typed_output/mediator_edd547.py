import itertools
import operator
import random
from fractions import Fraction
from typing import Callable, Any

from raiden.exceptions import UndefinedMediationFee
from raiden.transfer import channel, routes, secret_registry
from raiden.transfer.architecture import Event, StateChange, SuccessOrError, TransitionResult
from raiden.transfer.channel import get_balance
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE
from raiden.transfer.mediated_transfer.events import (
    EventUnexpectedSecretReveal,
    EventUnlockClaimFailed,
    EventUnlockClaimSuccess,
    EventUnlockFailed,
    EventUnlockSuccess,
    SendLockedTransfer,
    SendSecretReveal,
)
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState, Interpolate
from raiden.transfer.mediated_transfer.state import (
    LockedTransferSignedState,
    LockedTransferUnsignedState,
    MediationPairState,
    MediatorTransferState,
    WaitingTransferState,
)
from raiden.transfer.mediated_transfer.state_change import (
    ActionInitMediator,
    ReceiveLockExpired,
    ReceiveSecretReveal,
    ReceiveTransferRefund,
)
from raiden.transfer.state import (
    ChannelState,
    NettingChannelState,
    get_address_metadata,
    message_identifier_from_prng,
)
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, ReceiveUnlock
from raiden.transfer.utils import is_valid_secret_reveal
from raiden.utils.typing import (
    MYPY_ANNOTATION,
    Address,
    BlockExpiration,
    BlockHash,
    BlockNumber,
    BlockTimeout,
    ChannelID,
    Dict,
    List,
    LockType,
    Optional,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    TokenAmount,
    TokenNetworkAddress,
    Tuple,
    Union,
    cast,
    typecheck,
)

STATE_SECRET_KNOWN = (
    "payee_secret_revealed",
    "payee_contract_unlock",
    "payee_balance_proof",
    "payer_secret_revealed",
    "payer_waiting_unlock",
    "payer_balance_proof",
)
STATE_TRANSFER_PAID = ("payee_contract_unlock", "payee_balance_proof", "payer_balance_proof")
# TODO: fix expired state, it is not final
STATE_TRANSFER_FINAL = (
    "payee_contract_unlock",
    "payee_balance_proof",
    "payee_expired",
    "payer_balance_proof",
    "payer_expired",
)


def is_lock_valid(expiration: BlockExpiration, block_number: BlockNumber) -> bool:
    """True if the lock has not expired."""
    return block_number <= BlockNumber(expiration)


def is_safe_to_wait(
    lock_expiration: BlockExpiration,
    reveal_timeout: BlockTimeout,
    block_number: BlockNumber,
) -> SuccessOrError:
    """True if waiting is safe, i.e. there are more than enough blocks to safely
    unlock on chain.
    """
    # reveal timeout will not ever be larger than the lock_expiration otherwise
    # the expected block_number is negative
    assert block_number > 0, "block_number must be larger than zero"
    assert reveal_timeout > 0, "reveal_timeout must be larger than zero"
    assert lock_expiration > reveal_timeout, "lock_expiration must be larger than reveal_timeout"

    lock_timeout = lock_expiration - block_number

    # A node may wait for a new balance proof while there are reveal_timeout
    # blocks left, at that block and onwards it is not safe to wait.
    if lock_timeout > reveal_timeout:
        return SuccessOrError()

    return SuccessOrError(
        f"lock timeout is unsafe."
        f" timeout must be larger than {reveal_timeout}, but it is {lock_timeout}."
        f" expiration: {lock_expiration} block_number: {block_number}"
    )


def is_send_transfer_almost_equal(
    send_channel: NettingChannelState,  # pylint: disable=unused-argument
    send: LockedTransferUnsignedState,
    received: LockedTransferSignedState,
) -> bool:
    """True if both transfers are for the same mediated transfer."""
    # The only thing that may change is the direction of the transfer
    return (
        send.payment_identifier == received.payment_identifier
        and send.token == received.token
        # FIXME: Checking the transferred amount would make a lot of sense, but
        #        this is hard to do precisely without larger changes to the
        #        codebase. With the uncertainty about how we want to deal with
        #        refunds and backtracking in the long term, this check is
        #        skipped for now.
        # and send.lock.amount == received.lock.amount - send_channel.fee_schedule.flat
        and send.lock.expiration == received.lock.expiration
        and send.lock.secrethash == received.lock.secrethash
        and send.initiator == received.initiator
        and send.target == received.target
    )


def has_secret_registration_started(
    channel_states: List[NettingChannelState],
    transfers_pair: List[MediationPairState],
    secrethash: SecretHash,
) -> bool:
    # If it's known the secret is registered on-chain, the node should not send
    # a new transaction. Note there is a race condition:
    #
    # - Node B learns the secret on-chain, sends a secret reveal to A
    # - Node A receives the secret reveal off-chain prior to the event for the
    #   secret registration, if the lock is in the danger zone A will try to
    #   register the secret on-chain, because from its perspective the secret
    #   is not there yet.
    is_secret_registered_onchain = any(
        channel.is_secret_known_onchain(payer_channel.partner_state, secrethash)
        for payer_channel in channel_states
    )
    has_pending_transaction = any(
        pair.payer_state == "payer_waiting_secret_reveal" for pair in transfers_pair
    )
    return is_secret_registered_onchain or has_pending_transaction


def get_payee_channel(
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    transfer_pair: MediationPairState,
) -> Optional[NettingChannelState]:
    """Returns the payee channel of a given transfer pair or None if it's not found"""
    payee_channel_identifier = transfer_pair.payee_transfer.balance_proof.channel_identifier
    return channelidentifiers_to_channels.get(payee_channel_identifier)


def get_payer_channel(
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    transfer_pair: MediationPairState,
) -> Optional[NettingChannelState]:
    """Returns the payer channel of a given transfer pair or None if it's not found"""
    payer_channel_identifier = transfer_pair.payer_transfer.balance_proof.channel_identifier
    return channelidentifiers_to_channels.get(payer_channel_identifier)


def get_pending_transfer_pairs(
    transfers_pair: List[MediationPairState],
) -> List[MediationPairState]:
    """Return the transfer pairs that are not at a final state."""
    pending_pairs = [
        pair
        for pair in transfers_pair
        if pair.payee_state not in STATE_TRANSFER_FINAL
        or pair.payer_state not in STATE_TRANSFER_FINAL
    ]
    return pending_pairs


def find_intersection(
    fee_func: Interpolate,
    line: Callable[[int], Fraction],
) -> Optional[Fraction]:
    """Returns the x value where both functions intersect

    `fee_func` is a piecewise linear function while `line` is a straight line
    and takes the one of fee_func's indexes as argument.

    Returns `None` if there is no intersection within `fee_func`s domain, which
    indicates a lack of capacity.
    """
    i = 0
    y = fee_func.y_list[i]
    compare = operator.lt if y < line(i) else operator.gt
    while compare(y, line(i)):
        i += 1
        if i == len(fee_func.x_list):
            # Not enough capacity to send
            return None
        y = fee_func.y_list[i]

    # We found the linear section where the solution is. Now interpolate!
    x1 = fee_func.x_list[i - 1]
    x2 = fee_func.x_list[i]
    yf1 = fee_func.y_list[i - 1]
    yf2 = fee_func.y_list[i]
    yl1 = line(i - 1)
    yl2 = line(i)
    return (yl1 - yf1) * (x2 - x1) / ((yf2 - yf1) - (yl2 - yl1)) + x1


def get_amount_without_fees(
    amount_with_fees: PaymentWithFeeAmount,
    channel_in: NettingChannelState,
    channel_out: NettingChannelState,
) -> Optional[PaymentWithFeeAmount]:
    """Return the amount after fees are taken."""

    balance_in = get_balance(channel_in.our_state, channel_in.partner_state)
    balance_out = get_balance(channel_out.our_state, channel_out.partner_state)
    receivable = TokenAmount(
        channel_in.our_total_deposit + channel_in.partner_total_deposit - balance_in
    )
    assert (
        channel_in.fee_schedule.cap_fees == channel_out.fee_schedule.cap_fees
    ), "Both channels must have the same cap_fees setting for the same mediator."
    try:
        fee_func = FeeScheduleState.mediation_fee_func(
            schedule_in=channel_in.fee_schedule,
            schedule_out=channel_out.fee_schedule,
            balance_in=balance_in,
            balance_out=balance_out,
            receivable=receivable,
            amount_with_fees=amount_with_fees,
            cap_fees=channel_in.fee_schedule.cap_fees,
        )
        amount_without_fees = find_intersection(
            fee_func, lambda i: amount_with_fees - fee_func.x_list[i]
        )
    except UndefinedMediationFee:
        return None

    if amount_without_fees is None:
        # Insufficient capacity
        return None
    if amount_without_fees <= 0:
        # The node can't cover its mediations fees from the transferred amount.
        return None

    return PaymentWithFeeAmount(int(round(amount_without_fees)))


def sanity_check(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
) -> None:
    """Check invariants that must hold."""

    # if a transfer is paid we must know the secret
    all_transfers_states = itertools.chain(
        (pair.payee_state for pair in state.transfers_pair),
        (pair.payer_state for pair in state.transfers_pair),
    )
    if any(state in STATE_TRANSFER_PAID for state in all_transfers_states):
        assert state.secret is not None, "Mediator's state must have secret"

    # the "transitivity" for these values is checked below as part of
    # almost_equal check
    if state.transfers_pair:
        first_pair = state.transfers_pair[0]
        assert (
            state.secrethash == first_pair.payer_transfer.lock.secrethash
        ), "Secret hash mismatch"

    for pair in state.transfers_pair:
        payee_channel = get_payee_channel(
            channelidentifiers_to_channels=channelidentifiers_to_channels, transfer_pair=pair
        )

        # Channel could have been removed
        if not payee_channel:
            continue

        assert is_send_transfer_almost_equal(
            send_channel=payee_channel, send=pair.payee_transfer, received=pair.payer_transfer
        ), "Payee and payer transfers are too different"
        assert pair.payer_state in pair.valid_payer_states, "payer_state not in valid payer states"
        assert pair.payee_state in pair.valid_payee_states, "payee_state not in valid payee states"

    for original, refund in zip(state.transfers_pair[:-1], state.transfers_pair[1:]):
        assert original.payee_address == refund.payer_address, "payee/payer address mismatch"
        payer_channel = get_payer_channel(
            channelidentifiers_to_channels=channelidentifiers_to_channels, transfer_pair=refund
        )

        # Channel could have been removed
        if not payer_channel:
            continue

        transfer_sent = original.payee_transfer
        transfer_received = refund.payer_transfer
        assert is_send_transfer_almost_equal(
            send_channel=payer_channel, send=transfer_sent, received=transfer_received
        ), "Payee and payer transfers are too different (refund)"

    if state.waiting_transfer and state.transfers_pair:
        last_transfer_pair = state.transfers_pair[-1]
        payee_channel = get_payee_channel(
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            transfer_pair=last_transfer_pair,
        )
        # Channel could have been removed
        if payee_channel:
            transfer_sent = last_transfer_pair.payee_transfer
            transfer_received = state.waiting_transfer.transfer

            assert is_send_transfer_almost_equal(
                send_channel=payee_channel, send=transfer_sent, received=transfer_received
            ), "Payee and payer transfers are too different (waiting transfer)"


def clear_if_finalized(
    iteration: TransitionResult[Optional[MediatorTransferState]],
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
) -> TransitionResult[Optional[MediatorTransferState]]:
    """Clear the mediator task if all the locks have been finalized.

    A lock is considered finalized if it has been removed from the pending locks
    offchain, either because the transfer was unlocked or expired, or because the
    channel was settled on chain and therefore the channel is removed."""
    state = cast(MediatorTransferState, iteration.new_state)

    if state is None:
        return iteration  # type: ignore

    # Only clear the task if all channels have the lock cleared.
    secrethash = state.secrethash
    for pair in state.transfers_pair:
        payer_channel = get_payer_channel(channelidentifiers_to_channels, pair)
        if payer_channel and channel.is_lock_pending(payer_channel.partner_state, secrethash):
            return iteration

        payee_channel = get_payee_channel(channelidentifiers_to_channels, pair)
        if payee_channel and channel.is_lock_pending(payee_channel.our_state, secrethash):
            return iteration

    if state.waiting_transfer:
        waiting_transfer = state.waiting_transfer.transfer
        waiting_channel_identifier = waiting_transfer.balance_proof.channel_identifier
        waiting_channel = channelidentifiers_to_channels.get(waiting_channel_identifier)

        if waiting_channel and channel.is_lock_pending(waiting_channel.partner_state, secrethash):
            return iteration

    return TransitionResult(None, iteration.events)


def forward_transfer_pair(
    payer_transfer: LockedTransferSignedState,
    payer_channel: NettingChannelState,
    payee_channel: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> Tuple[Optional[MediationPairState], List[Event]]:
    """Given a payer transfer tries the given route to proceed with the mediation.

    Args:
        payer_transfer: The transfer received from the payer_channel.
        channelidentifiers_to_channels: All the channels available for this
            transfer.

        pseudo_random_generator: Number generator to generate a message id.
        block_number: The current block number.
    """

    amount_after_fees = get_amount_without_fees(
        amount_with_fees=payer_transfer.lock.amount,
        channel_in=payer_channel,
        channel_out=payee_channel,
    )
    if not amount_after_fees:
        return None, []

    lock_timeout = BlockTimeout(payer_transfer.lock.expiration - block_number)
    safe_to_use_channel = channel.is_channel_usable_for_mediation(
        channel_state=payee_channel, transfer_amount=amount_after_fees, lock_timeout=lock_timeout
    )

    if not safe_to_use_channel:
        return None, []

    assert payee_channel.settle_timeout >= lock_timeout, "settle_timeout must be >= lock_timeout"

    message_identifier = message_identifier_from_prng(pseudo_random_generator)

    recipient_address = payee_channel.partner_state.address
    recipient_metadata = get_address_metadata(recipient_address, payer_transfer.route_states)
    lockedtransfer_event = channel.send_lockedtransfer(
        channel_state=payee_channel,
        initiator=payer_transfer.initiator,
        target=payer_transfer.target,
        amount=amount_after_fees,
        message_identifier=message_identifier,
        payment_identifier=payer_transfer.payment_identifier,
        expiration=payer_transfer.lock.expiration,
        secret=payer_transfer.secret,
        secrethash=payer_transfer.lock.secrethash,
        route_states=payer_transfer.route_states,
        recipient_metadata=recipient_metadata,
        previous_metadata=payer_transfer.metadata,
    )
    mediated_events: List[Event] = [lockedtransfer_event]

    # create transfer pair
    transfer_pair = MediationPairState(
        payer_transfer=payer_transfer,
        payee_address=payee_channel.partner_state.address,
        payee_transfer=lockedtransfer_event.transfer,
    )

    return transfer_pair, mediated_events


def set_offchain_secret(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    secret: Secret,
    secrethash: SecretHash,
) -> List[Event]:
    """Set the secret to all mediated transfers."""
    state.secret = secret

    for pair in state.transfers_pair