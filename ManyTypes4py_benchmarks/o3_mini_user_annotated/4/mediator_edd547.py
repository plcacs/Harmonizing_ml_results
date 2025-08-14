#!/usr/bin/env python3
import itertools
import operator
import random
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    Dict as TDict,
    List as TList,
    LockType,
    Optional as TOptional,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    TokenAmount,
    TokenNetworkAddress,
    Tuple as TTuple,
    Union as TUnion,
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
    lock_expiration: BlockExpiration, reveal_timeout: BlockTimeout, block_number: BlockNumber
) -> SuccessOrError:
    """True if waiting is safe, i.e. there are more than enough blocks to safely
    unlock on chain.
    """
    assert block_number > 0, "block_number must be larger than zero"
    assert reveal_timeout > 0, "reveal_timeout must be larger than zero"
    assert lock_expiration > reveal_timeout, "lock_expiration must be larger than reveal_timeout"

    lock_timeout = lock_expiration - block_number

    if lock_timeout > reveal_timeout:
        return SuccessOrError()

    return SuccessOrError(
        f"lock timeout is unsafe."
        f" timeout must be larger than {reveal_timeout}, but it is {lock_timeout}."
        f" expiration: {lock_expiration} block_number: {block_number}"
    )


def is_send_transfer_almost_equal(
    send_channel: NettingChannelState,
    send: LockedTransferUnsignedState,
    received: LockedTransferSignedState,
) -> bool:
    """True if both transfers are for the same mediated transfer."""
    return (
        send.payment_identifier == received.payment_identifier
        and send.token == received.token
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
    is_secret_registered_onchain: bool = any(
        channel.is_secret_known_onchain(payer_channel.partner_state, secrethash)
        for payer_channel in channel_states
    )
    has_pending_transaction: bool = any(
        pair.payer_state == "payer_waiting_secret_reveal" for pair in transfers_pair
    )
    return is_secret_registered_onchain or has_pending_transaction


def get_payee_channel(
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    transfer_pair: MediationPairState,
) -> Optional[NettingChannelState]:
    payee_channel_identifier: ChannelID = transfer_pair.payee_transfer.balance_proof.channel_identifier
    return channelidentifiers_to_channels.get(payee_channel_identifier)


def get_payer_channel(
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    transfer_pair: MediationPairState,
) -> Optional[NettingChannelState]:
    payer_channel_identifier: ChannelID = transfer_pair.payer_transfer.balance_proof.channel_identifier
    return channelidentifiers_to_channels.get(payer_channel_identifier)


def get_pending_transfer_pairs(
    transfers_pair: List[MediationPairState],
) -> List[MediationPairState]:
    pending_pairs: List[MediationPairState] = [
        pair
        for pair in transfers_pair
        if pair.payee_state not in STATE_TRANSFER_FINAL or pair.payer_state not in STATE_TRANSFER_FINAL
    ]
    return pending_pairs


def find_intersection(
    fee_func: Interpolate, line: Callable[[int], Fraction]
) -> Optional[Fraction]:
    """Returns the x value where both functions intersect"""
    i: int = 0
    y: Fraction = fee_func.y_list[i]
    compare = operator.lt if y < line(i) else operator.gt
    while compare(y, line(i)):
        i += 1
        if i == len(fee_func.x_list):
            return None
        y = fee_func.y_list[i]
    x1: int = fee_func.x_list[i - 1]
    x2: int = fee_func.x_list[i]
    yf1: Fraction = fee_func.y_list[i - 1]
    yf2: Fraction = fee_func.y_list[i]
    yl1: Fraction = line(i - 1)
    yl2: Fraction = line(i)
    return (yl1 - yf1) * (x2 - x1) / ((yf2 - yf1) - (yl2 - yl1)) + x1


def get_amount_without_fees(
    amount_with_fees: PaymentWithFeeAmount,
    channel_in: NettingChannelState,
    channel_out: NettingChannelState,
) -> Optional[PaymentWithFeeAmount]:
    balance_in: TokenAmount = get_balance(channel_in.our_state, channel_in.partner_state)
    balance_out: TokenAmount = get_balance(channel_out.our_state, channel_out.partner_state)
    receivable: TokenAmount = TokenAmount(
        channel_in.our_total_deposit + channel_in.partner_total_deposit - balance_in
    )
    assert (
        channel_in.fee_schedule.cap_fees == channel_out.fee_schedule.cap_fees
    ), "Both channels must have the same cap_fees setting for the same mediator."
    try:
        fee_func: Interpolate = FeeScheduleState.mediation_fee_func(
            schedule_in=channel_in.fee_schedule,
            schedule_out=channel_out.fee_schedule,
            balance_in=balance_in,
            balance_out=balance_out,
            receivable=receivable,
            amount_with_fees=amount_with_fees,
            cap_fees=channel_in.fee_schedule.cap_fees,
        )
        amount_without_fees: Optional[Fraction] = find_intersection(
            fee_func, lambda i: amount_with_fees - fee_func.x_list[i]
        )
    except UndefinedMediationFee:
        return None

    if amount_without_fees is None:
        return None
    if amount_without_fees <= 0:
        return None

    return PaymentWithFeeAmount(int(round(amount_without_fees)))


def sanity_check(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
) -> None:
    all_transfers_states = itertools.chain(
        (pair.payee_state for pair in state.transfers_pair),
        (pair.payer_state for pair in state.transfers_pair),
    )
    if any(state_ in STATE_TRANSFER_PAID for state_ in all_transfers_states):
        assert state.secret is not None, "Mediator's state must have secret"

    if state.transfers_pair:
        first_pair: MediationPairState = state.transfers_pair[0]
        assert (
            state.secrethash == first_pair.payer_transfer.lock.secrethash
        ), "Secret hash mismatch"

    for pair in state.transfers_pair:
        payee_channel: Optional[NettingChannelState] = get_payee_channel(
            channelidentifiers_to_channels=channelidentifiers_to_channels, transfer_pair=pair
        )
        if not payee_channel:
            continue

        assert is_send_transfer_almost_equal(
            send_channel=payee_channel, send=pair.payee_transfer, received=pair.payer_transfer
        ), "Payee and payer transfers are too different"
        assert pair.payer_state in pair.valid_payer_states, "payer_state not in valid payer states"
        assert pair.payee_state in pair.valid_payee_states, "payee_state not in valid payee states"

    for original, refund in zip(state.transfers_pair[:-1], state.transfers_pair[1:]):
        assert original.payee_address == refund.payer_address, "payee/payer address mismatch"
        payer_channel: Optional[NettingChannelState] = get_payer_channel(
            channelidentifiers_to_channels=channelidentifiers_to_channels, transfer_pair=refund
        )
        if not payer_channel:
            continue

        transfer_sent = original.payee_transfer
        transfer_received = refund.payer_transfer
        assert is_send_transfer_almost_equal(
            send_channel=payer_channel, send=transfer_sent, received=transfer_received
        ), "Payee and payer transfers are too different (refund)"

    if state.waiting_transfer and state.transfers_pair:
        last_transfer_pair: MediationPairState = state.transfers_pair[-1]
        payee_channel = get_payee_channel(
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            transfer_pair=last_transfer_pair,
        )
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
    state: Optional[MediatorTransferState] = cast(MediatorTransferState, iteration.new_state)

    if state is None:
        return iteration  # type: ignore

    secrethash: SecretHash = state.secrethash
    for pair in state.transfers_pair:
        payer_channel: Optional[NettingChannelState] = get_payer_channel(channelidentifiers_to_channels, pair)
        if payer_channel and channel.is_lock_pending(payer_channel.partner_state, secrethash):
            return iteration

        payee_channel: Optional[NettingChannelState] = get_payee_channel(channelidentifiers_to_channels, pair)
        if payee_channel and channel.is_lock_pending(payee_channel.our_state, secrethash):
            return iteration

    if state.waiting_transfer:
        waiting_transfer = state.waiting_transfer.transfer
        waiting_channel_identifier: ChannelID = waiting_transfer.balance_proof.channel_identifier
        waiting_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(waiting_channel_identifier)

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
    amount_after_fees: Optional[PaymentWithFeeAmount] = get_amount_without_fees(
        amount_with_fees=payer_transfer.lock.amount,
        channel_in=payer_channel,
        channel_out=payee_channel,
    )
    if not amount_after_fees:
        return None, []

    lock_timeout: BlockTimeout = BlockTimeout(payer_transfer.lock.expiration - block_number)
    safe_to_use_channel: bool = channel.is_channel_usable_for_mediation(
        channel_state=payee_channel, transfer_amount=amount_after_fees, lock_timeout=lock_timeout
    )

    if not safe_to_use_channel:
        return None, []

    assert payee_channel.settle_timeout >= lock_timeout, "settle_timeout must be >= lock_timeout"

    message_identifier = message_identifier_from_prng(pseudo_random_generator)

    recipient_address: Address = payee_channel.partner_state.address
    recipient_metadata = get_address_metadata(recipient_address, payer_transfer.route_states)
    lockedtransfer_event: SendLockedTransfer = channel.send_lockedtransfer(
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

    transfer_pair: MediationPairState = MediationPairState(
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
    state.secret = secret

    for pair in state.transfers_pair:
        payer_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(
            pair.payer_transfer.balance_proof.channel_identifier
        )
        if payer_channel:
            channel.register_offchain_secret(payer_channel, secret, secrethash)

        payee_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(
            pair.payee_transfer.balance_proof.channel_identifier
        )
        if payee_channel:
            channel.register_offchain_secret(payee_channel, secret, secrethash)

    if state.waiting_transfer:
        payer_channel = channelidentifiers_to_channels.get(
            state.waiting_transfer.transfer.balance_proof.channel_identifier
        )
        if payer_channel:
            channel.register_offchain_secret(payer_channel, secret, secrethash)

        unexpected_reveal: EventUnexpectedSecretReveal = EventUnexpectedSecretReveal(
            secrethash=secrethash, reason="The mediator has a waiting transfer."
        )
        return [unexpected_reveal]

    return []


def set_onchain_secret(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    secret: Secret,
    secrethash: SecretHash,
    block_number: BlockNumber,
) -> List[Event]:
    state.secret = secret

    for pair in state.transfers_pair:
        payer_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(
            pair.payer_transfer.balance_proof.channel_identifier
        )
        if payer_channel:
            channel.register_onchain_secret(payer_channel, secret, secrethash, block_number)

        payee_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(
            pair.payee_transfer.balance_proof.channel_identifier
        )
        if payee_channel:
            channel.register_onchain_secret(
                channel_state=payee_channel,
                secret=secret,
                secrethash=secrethash,
                secret_reveal_block_number=block_number,
            )

    if state.waiting_transfer:
        payer_channel = channelidentifiers_to_channels.get(
            state.waiting_transfer.transfer.balance_proof.channel_identifier
        )
        if payer_channel:
            channel.register_onchain_secret(
                channel_state=payer_channel,
                secret=secret,
                secrethash=secrethash,
                secret_reveal_block_number=block_number,
            )

        unexpected_reveal: EventUnexpectedSecretReveal = EventUnexpectedSecretReveal(
            secrethash=secrethash, reason="The mediator has a waiting transfer."
        )
        return [unexpected_reveal]

    return []


def set_offchain_reveal_state(
    transfers_pair: List[MediationPairState], payee_address: Address
) -> None:
    for pair in transfers_pair:
        if pair.payee_address == payee_address:
            pair.payee_state = "payee_secret_revealed"


def events_for_expired_pairs(
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    transfers_pair: List[MediationPairState],
    waiting_transfer: Optional[WaitingTransferState],
    block_number: BlockNumber,
) -> List[Event]:
    pending_transfers_pairs: List[MediationPairState] = get_pending_transfer_pairs(transfers_pair)
    events: List[Event] = []
    for pair in pending_transfers_pairs:
        payer_balance_proof = pair.payer_transfer.balance_proof
        payer_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(payer_balance_proof.channel_identifier)
        if not payer_channel:
            continue

        has_payer_transfer_expired: bool = channel.is_transfer_expired(
            transfer=pair.payer_transfer, affected_channel=payer_channel, block_number=block_number
        )

        if has_payer_transfer_expired:
            pair.payer_state = "payer_expired"
            unlock_claim_failed: EventUnlockClaimFailed = EventUnlockClaimFailed(
                pair.payer_transfer.payment_identifier,
                pair.payer_transfer.lock.secrethash,
                "lock expired",
            )
            events.append(unlock_claim_failed)

    if waiting_transfer:
        expiration_threshold = channel.get_receiver_expiration_threshold(
            waiting_transfer.transfer.lock.expiration
        )
        should_waiting_transfer_expire: bool = (
            waiting_transfer.state != "expired" and expiration_threshold <= block_number
        )
        if should_waiting_transfer_expire:
            waiting_transfer.state = "expired"
            unlock_claim_failed: EventUnlockClaimFailed = EventUnlockClaimFailed(
                waiting_transfer.transfer.payment_identifier,
                waiting_transfer.transfer.lock.secrethash,
                "lock expired",
            )
            events.append(unlock_claim_failed)

    return events


def events_for_secretreveal(
    transfers_pair: List[MediationPairState],
    secret: Secret,
    pseudo_random_generator: random.Random,
) -> List[Event]:
    events: List[Event] = []
    for pair in reversed(transfers_pair):
        payee_knows_secret: bool = pair.payee_state in STATE_SECRET_KNOWN
        payer_knows_secret: bool = pair.payer_state in STATE_SECRET_KNOWN
        is_transfer_pending: bool = pair.payer_state == "payer_pending"

        should_send_secret: bool = payee_knows_secret and not payer_knows_secret and is_transfer_pending

        if should_send_secret:
            message_identifier = message_identifier_from_prng(pseudo_random_generator)
            pair.payer_state = "payer_secret_revealed"
            payer_transfer = pair.payer_transfer
            revealsecret: SendSecretReveal = SendSecretReveal(
                recipient=payer_transfer.balance_proof.sender,
                recipient_metadata=payer_transfer.payer_address_metadata,
                message_identifier=message_identifier,
                secret=secret,
                canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE,
            )

            events.append(revealsecret)
    return events


def events_for_balanceproof(
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    transfers_pair: List[MediationPairState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    secret: Secret,
    secrethash: SecretHash,
) -> List[Event]:
    events: List[Event] = []
    for pair in reversed(transfers_pair):
        payee_knows_secret: bool = pair.payee_state in STATE_SECRET_KNOWN
        payee_payed: bool = pair.payee_state in STATE_TRANSFER_PAID

        payee_channel: Optional[NettingChannelState] = get_payee_channel(channelidentifiers_to_channels, pair)
        payee_channel_open: bool = (
            payee_channel is not None and channel.get_status(payee_channel) == ChannelState.STATE_OPENED
        )

        payer_channel: Optional[NettingChannelState] = get_payer_channel(channelidentifiers_to_channels, pair)

        is_safe_to_send_balanceproof: bool = False
        if payer_channel:
            is_safe_to_send_balanceproof = is_safe_to_wait(
                pair.payer_transfer.lock.expiration, payer_channel.reveal_timeout, block_number
            ).ok

        should_send_balanceproof_to_payee: bool = (
            payee_channel_open
            and payee_knows_secret
            and not payee_payed
            and is_safe_to_send_balanceproof
        )

        if should_send_balanceproof_to_payee:
            assert payee_channel, MYPY_ANNOTATION
            pair.payee_state = "payee_balance_proof"

            message_identifier = message_identifier_from_prng(pseudo_random_generator)
            recipient_address: Address = pair.payee_address
            recipient_metadata = get_address_metadata(
                recipient_address, pair.payee_transfer.route_states
            )
            unlock_lock: Event = channel.send_unlock(
                channel_state=payee_channel,
                message_identifier=message_identifier,
                payment_identifier=pair.payee_transfer.payment_identifier,
                secret=secret,
                secrethash=secrethash,
                block_number=block_number,
                recipient_metadata=recipient_metadata,
            )

            unlock_success: EventUnlockSuccess = EventUnlockSuccess(
                pair.payer_transfer.payment_identifier, pair.payer_transfer.lock.secrethash
            )
            events.append(unlock_lock)
            events.append(unlock_success)
    return events


def events_for_onchain_secretreveal_if_dangerzone(
    channelmap: Dict[ChannelID, NettingChannelState],
    secrethash: SecretHash,
    transfers_pair: List[MediationPairState],
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[Event]:
    events: List[Event] = []
    all_payer_channels: List[NettingChannelState] = []
    for pair in transfers_pair:
        channel_state: Optional[NettingChannelState] = get_payer_channel(channelmap, pair)
        if channel_state:
            all_payer_channels.append(channel_state)
    transaction_sent: bool = has_secret_registration_started(
        all_payer_channels, transfers_pair, secrethash
    )

    for pair in get_pending_transfer_pairs(transfers_pair):
        payer_channel: Optional[NettingChannelState] = get_payer_channel(channelmap, pair)
        if not payer_channel:
            continue

        lock = pair.payer_transfer.lock
        safe_to_wait = is_safe_to_wait(lock.expiration, payer_channel.reveal_timeout, block_number)
        secret_known: bool = channel.is_secret_known(
            payer_channel.partner_state, pair.payer_transfer.lock.secrethash
        )

        if not safe_to_wait and secret_known:
            pair.payer_state = "payer_waiting_secret_reveal"

            if not transaction_sent:
                secret: Optional[Secret] = channel.get_secret(payer_channel.partner_state, lock.secrethash)
                assert secret, "the secret should be known at this point"

                reveal_events: List[Event] = secret_registry.events_for_onchain_secretreveal(
                    channel_state=payer_channel,
                    secret=secret,
                    expiration=lock.expiration,
                    block_hash=block_hash,
                )
                events.extend(reveal_events)
                transaction_sent = True
    return events


def events_for_onchain_secretreveal_if_closed(
    channelmap: Dict[ChannelID, NettingChannelState],
    transfers_pair: List[MediationPairState],
    secret: Secret,
    secrethash: SecretHash,
    block_hash: BlockHash,
) -> List[Event]:
    events: List[Event] = []
    all_payer_channels: List[NettingChannelState] = []
    for pair in transfers_pair:
        channel_state: Optional[NettingChannelState] = get_payer_channel(channelmap, pair)
        if channel_state:
            all_payer_channels.append(channel_state)
    transaction_sent: bool = has_secret_registration_started(
        all_payer_channels, transfers_pair, secrethash
    )

    for pending_pair in get_pending_transfer_pairs(transfers_pair):
        payer_channel: Optional[NettingChannelState] = get_payer_channel(channelmap, pending_pair)
        if payer_channel and channel.get_status(payer_channel) == ChannelState.STATE_CLOSED:
            pending_pair.payer_state = "payer_waiting_secret_reveal"

            if not transaction_sent:
                partner_state = payer_channel.partner_state
                lock = channel.get_lock(partner_state, secrethash)
                if lock:
                    reveal_events: List[Event] = secret_registry.events_for_onchain_secretreveal(
                        channel_state=payer_channel,
                        secret=secret,
                        expiration=lock.expiration,
                        block_hash=block_hash,
                    )
                    events.extend(reveal_events)
                    transaction_sent = True
    return events


def events_to_remove_expired_locks(
    mediator_state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    block_number: BlockNumber,
    pseudo_random_generator: random.Random,
) -> List[Event]:
    events: List[Event] = []
    if not mediator_state.transfers_pair:
        return events

    initial_payer_transfer: LockedTransferSignedState = mediator_state.transfers_pair[0].payer_transfer
    for transfer_pair in mediator_state.transfers_pair:
        balance_proof = transfer_pair.payee_transfer.balance_proof
        channel_identifier: ChannelID = balance_proof.channel_identifier
        channel_state: Optional[NettingChannelState] = channelidentifiers_to_channels.get(channel_identifier)
        if not channel_state:
            continue

        secrethash: SecretHash = mediator_state.secrethash
        lock: Optional[LockType] = None
        if secrethash in channel_state.our_state.secrethashes_to_lockedlocks:
            assert secrethash not in channel_state.our_state.secrethashes_to_unlockedlocks, "Locks for secrethash are already unlocked"
            lock = channel_state.our_state.secrethashes_to_lockedlocks.get(secrethash)
        elif secrethash in channel_state.our_state.secrethashes_to_unlockedlocks:
            lock = channel_state.our_state.secrethashes_to_unlockedlocks.get(secrethash)
        if lock:
            lock_expiration_threshold = channel.get_sender_expiration_threshold(lock.expiration)
            has_lock_expired: bool = channel.is_lock_expired(
                end_state=channel_state.our_state,
                lock=lock,
                block_number=block_number,
                lock_expiration_threshold=lock_expiration_threshold,
            )
            is_channel_open: bool = channel.get_status(channel_state) == ChannelState.STATE_OPENED
            payee_address_metadata = get_address_metadata(
                transfer_pair.payee_address, initial_payer_transfer.route_states
            )
            if has_lock_expired and is_channel_open:
                transfer_pair.payee_state = "payee_expired"
                expired_lock_events: List[Event] = channel.send_lock_expired(
                    channel_state=channel_state,
                    locked_lock=lock,
                    pseudo_random_generator=pseudo_random_generator,
                    recipient_metadata=payee_address_metadata,
                )
                events.extend(expired_lock_events)

                unlock_failed = EventUnlockFailed(
                    transfer_pair.payee_transfer.payment_identifier,
                    transfer_pair.payee_transfer.lock.secrethash,
                    "lock expired",
                )
                events.append(unlock_failed)
    return events


def secret_learned(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    block_hash: BlockHash,
    secret: Secret,
    secrethash: SecretHash,
    payee_address: Address,
) -> TransitionResult[MediatorTransferState]:
    secret_reveal_events: List[Event] = set_offchain_secret(
        state, channelidentifiers_to_channels, secret, secrethash
    )

    set_offchain_reveal_state(state.transfers_pair, payee_address)

    onchain_secret_reveal: List[Event] = events_for_onchain_secretreveal_if_closed(
        channelmap=channelidentifiers_to_channels,
        transfers_pair=state.transfers_pair,
        secret=secret,
        secrethash=secrethash,
        block_hash=block_hash,
    )

    offchain_secret_reveal: List[Event] = events_for_secretreveal(
        state.transfers_pair, secret, pseudo_random_generator
    )

    balance_proof: List[Event] = events_for_balanceproof(
        channelidentifiers_to_channels,
        state.transfers_pair,
        pseudo_random_generator,
        block_number,
        secret,
        secrethash,
    )

    events: List[Event] = secret_reveal_events + offchain_secret_reveal + balance_proof + onchain_secret_reveal
    iteration: TransitionResult[MediatorTransferState] = TransitionResult(state, events)
    return iteration


def mediate_transfer(
    state: MediatorTransferState,
    payer_channel: NettingChannelState,
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], NettingChannelState],
    pseudo_random_generator: random.Random,
    payer_transfer: LockedTransferSignedState,
    block_number: BlockNumber,
) -> TransitionResult[MediatorTransferState]:
    assert (
        payer_channel.partner_state.address == payer_transfer.balance_proof.sender
    ), "Transfer must be signed by sender"

    our_address: Address = payer_channel.our_state.address
    candidate_route_states = routes.filter_acceptable_routes(
        route_states=state.routes,
        blacklisted_channel_ids=state.refunded_channels,
        addresses_to_channel=addresses_to_channel,
        token_network_address=payer_channel.token_network_address,
        our_address=our_address,
    )

    for route_state in candidate_route_states:
        next_hop: Optional[Address] = route_state.hop_after(our_address)
        if not next_hop:
            continue
        target_token_network: TokenNetworkAddress = route_state.swaps.get(
            our_address, payer_channel.token_network_address
        )
        payee_channel: Optional[NettingChannelState] = addresses_to_channel.get((target_token_network, next_hop))
        if not payee_channel:
            continue

        mediation_transfer_pair, mediation_events = forward_transfer_pair(
            payer_transfer=payer_transfer,
            payer_channel=payer_channel,
            payee_channel=payee_channel,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number,
        )
        if mediation_transfer_pair is not None:
            state.transfers_pair.append(mediation_transfer_pair)
            return TransitionResult(state, mediation_events)

    state.waiting_transfer = WaitingTransferState(payer_transfer)
    return TransitionResult(state, [])


def handle_init(
    state_change: ActionInitMediator,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[MediatorTransferState]]:
    from_hop = state_change.from_hop
    from_transfer = state_change.from_transfer
    payer_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(from_hop.channel_identifier)

    if not payer_channel:
        return TransitionResult(None, [])

    mediator_state: MediatorTransferState = MediatorTransferState(
        secrethash=from_transfer.lock.secrethash, routes=state_change.candidate_route_states
    )

    is_valid, events, _ = channel.handle_receive_lockedtransfer(
        payer_channel,
        from_transfer,
        recipient_metadata=state_change.from_transfer.payer_address_metadata,
    )
    if not is_valid:
        return TransitionResult(None, events)

    iteration: TransitionResult[MediatorTransferState] = mediate_transfer(
        state=mediator_state,
        payer_channel=payer_channel,
        addresses_to_channel=addresses_to_channel,
        pseudo_random_generator=pseudo_random_generator,
        payer_transfer=from_transfer,
        block_number=block_number,
    )

    events.extend(iteration.events)
    return TransitionResult(iteration.new_state, events)


def handle_block(
    mediator_state: MediatorTransferState,
    state_change: Block,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], NettingChannelState],
    pseudo_random_generator: random.Random,
) -> TransitionResult[MediatorTransferState]:
    mediate_events: List[Event] = []
    if mediator_state.waiting_transfer:
        secrethash: SecretHash = mediator_state.waiting_transfer.transfer.lock.secrethash
        payer_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(
            mediator_state.waiting_transfer.transfer.balance_proof.channel_identifier
        )
        if payer_channel is not None:
            mediation_attempt: TransitionResult[MediatorTransferState] = mediate_transfer(
                state=mediator_state,
                payer_channel=payer_channel,
                addresses_to_channel=addresses_to_channel,
                pseudo_random_generator=pseudo_random_generator,
                payer_transfer=mediator_state.waiting_transfer.transfer,
                block_number=state_change.block_number,
            )
            mediator_state = mediation_attempt.new_state  # type: ignore
            mediate_events = mediation_attempt.events
            success_filter = lambda event: (
                isinstance(event, SendLockedTransfer)
                and event.transfer.lock.secrethash == secrethash
            )

            mediation_happened: bool = any(filter(success_filter, mediate_events))
            if mediation_happened:
                mediator_state.waiting_transfer = None

    expired_locks_events: List[Event] = events_to_remove_expired_locks(
        mediator_state=mediator_state,
        channelidentifiers_to_channels=channelidentifiers_to_channels,
        block_number=state_change.block_number,
        pseudo_random_generator=pseudo_random_generator,
    )

    secret_reveal_events: List[Event] = events_for_onchain_secretreveal_if_dangerzone(
        channelmap=channelidentifiers_to_channels,
        secrethash=mediator_state.secrethash,
        transfers_pair=mediator_state.transfers_pair,
        block_number=state_change.block_number,
        block_hash=state_change.block_hash,
    )

    unlock_fail_events: List[Event] = events_for_expired_pairs(
        channelidentifiers_to_channels=channelidentifiers_to_channels,
        transfers_pair=mediator_state.transfers_pair,
        waiting_transfer=mediator_state.waiting_transfer,
        block_number=state_change.block_number,
    )

    iteration: TransitionResult[MediatorTransferState] = TransitionResult(
        mediator_state,
        mediate_events + unlock_fail_events + secret_reveal_events + expired_locks_events,
    )
    return iteration


def handle_refundtransfer(
    mediator_state: MediatorTransferState,
    mediator_state_change: ReceiveTransferRefund,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[MediatorTransferState]:
    events: List[Event] = []
    if mediator_state.secret is None:
        transfer_pair: MediationPairState = mediator_state.transfers_pair[-1]
        payee_transfer = transfer_pair.payee_transfer
        payer_transfer = mediator_state_change.transfer
        channel_identifier: ChannelID = payer_transfer.balance_proof.channel_identifier
        payer_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(channel_identifier)
        if not payer_channel:
            return TransitionResult(mediator_state, [])

        is_valid, channel_events, _ = channel.handle_refundtransfer(
            received_transfer=payee_transfer,
            channel_state=payer_channel,
            refund=mediator_state_change,
        )
        if not is_valid:
            return TransitionResult(mediator_state, channel_events)

        mediator_state.refunded_channels.append(
            payer_channel.canonical_identifier.channel_identifier
        )
        iteration: TransitionResult[MediatorTransferState] = mediate_transfer(
            state=mediator_state,
            payer_channel=payer_channel,
            addresses_to_channel=addresses_to_channel,
            pseudo_random_generator=pseudo_random_generator,
            payer_transfer=payer_transfer,
            block_number=block_number,
        )
        events.extend(channel_events)
        events.extend(iteration.events)

    iteration: TransitionResult[MediatorTransferState] = TransitionResult(mediator_state, events)
    return iteration


def handle_offchain_secretreveal(
    mediator_state: MediatorTransferState,
    mediator_state_change: ReceiveSecretReveal,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> TransitionResult[MediatorTransferState]:
    is_valid_reveal: bool = is_valid_secret_reveal(
        state_change=mediator_state_change, transfer_secrethash=mediator_state.secrethash
    )
    is_secret_unknown: bool = mediator_state.secret is None

    if not mediator_state.transfers_pair:
        return TransitionResult(mediator_state, [])

    transfer_pair: MediationPairState = mediator_state.transfers_pair[-1]
    payer_transfer = transfer_pair.payer_transfer
    channel_identifier: ChannelID = payer_transfer.balance_proof.channel_identifier
    payer_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(channel_identifier)
    if not payer_channel:
        return TransitionResult(mediator_state, [])

    has_payer_transfer_expired: bool = channel.is_transfer_expired(
        transfer=transfer_pair.payer_transfer,
        affected_channel=payer_channel,
        block_number=block_number,
    )

    if is_secret_unknown and is_valid_reveal and not has_payer_transfer_expired:
        iteration: TransitionResult[MediatorTransferState] = secret_learned(
            state=mediator_state,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number,
            block_hash=block_hash,
            secret=mediator_state_change.secret,
            secrethash=mediator_state_change.secrethash,
            payee_address=mediator_state_change.sender,
        )
    else:
        iteration = TransitionResult(mediator_state, [])
    return iteration


def handle_onchain_secretreveal(
    mediator_state: MediatorTransferState,
    onchain_secret_reveal: ContractReceiveSecretReveal,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[MediatorTransferState]:
    secrethash: SecretHash = onchain_secret_reveal.secrethash
    is_valid_reveal: bool = is_valid_secret_reveal(
        state_change=onchain_secret_reveal, transfer_secrethash=mediator_state.secrethash
    )
    if is_valid_reveal:
        secret: Secret = onchain_secret_reveal.secret
        block_number = onchain_secret_reveal.block_number
        secret_reveal: List[Event] = set_onchain_secret(
            state=mediator_state,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            secret=secret,
            secrethash=secrethash,
            block_number=block_number,
        )
        balance_proof: List[Event] = events_for_balanceproof(
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            transfers_pair=mediator_state.transfers_pair,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number,
            secret=secret,
            secrethash=secrethash,
        )
        iteration: TransitionResult[MediatorTransferState] = TransitionResult(mediator_state, secret_reveal + balance_proof)
    else:
        iteration = TransitionResult(mediator_state, [])
    return iteration


def handle_unlock(
    mediator_state: MediatorTransferState,
    state_change: ReceiveUnlock,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
) -> TransitionResult[MediatorTransferState]:
    events: List[Event] = []
    balance_proof_sender: Address = state_change.balance_proof.sender
    channel_identifier: ChannelID = state_change.balance_proof.channel_identifier

    for pair in mediator_state.transfers_pair:
        if pair.payer_transfer.balance_proof.sender == balance_proof_sender:
            channel_state: Optional[NettingChannelState] = channelidentifiers_to_channels.get(channel_identifier)
            if channel_state:
                recipient_metadata = get_address_metadata(
                    balance_proof_sender, mediator_state.routes
                )
                is_valid, channel_events, _ = channel.handle_unlock(
                    channel_state, state_change, recipient_metadata
                )
                events.extend(channel_events)
                if is_valid:
                    unlock: EventUnlockClaimSuccess = EventUnlockClaimSuccess(
                        pair.payee_transfer.payment_identifier, pair.payee_transfer.lock.secrethash
                    )
                    events.append(unlock)
                    pair.payer_state = "payer_balance_proof"
    iteration: TransitionResult[MediatorTransferState] = TransitionResult(mediator_state, events)
    return iteration


def handle_lock_expired(
    mediator_state: MediatorTransferState,
    state_change: ReceiveLockExpired,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    block_number: BlockNumber,
) -> TransitionResult[MediatorTransferState]:
    events: List[Event] = []
    for transfer_pair in mediator_state.transfers_pair:
        balance_proof = transfer_pair.payer_transfer.balance_proof
        channel_state: Optional[NettingChannelState] = channelidentifiers_to_channels.get(balance_proof.channel_identifier)
        if not channel_state:
            return TransitionResult(mediator_state, [])

        recipient_address: Address = channel_state.partner_state.address
        recipient_metadata = get_address_metadata(recipient_address, mediator_state.routes)
        result = channel.handle_receive_lock_expired(
            channel_state=channel_state,
            state_change=state_change,
            block_number=block_number,
            recipient_metadata=recipient_metadata,
        )
        assert result.new_state and isinstance(result.new_state, NettingChannelState), (
            "Handling a receive_lock_expire should never delete the channel task",
        )
        events.extend(result.events)
        if not channel.get_lock(result.new_state.partner_state, mediator_state.secrethash):
            transfer_pair.payer_state = "payer_expired"

    if mediator_state.waiting_transfer:
        waiting_channel: Optional[NettingChannelState] = channelidentifiers_to_channels.get(
            mediator_state.waiting_transfer.transfer.balance_proof.channel_identifier
        )
        if waiting_channel:
            recipient_address = waiting_channel.partner_state.address
            recipient_metadata = get_address_metadata(recipient_address, mediator_state.routes)
            result = channel.handle_receive_lock_expired(
                channel_state=waiting_channel,
                state_change=state_change,
                block_number=block_number,
                recipient_metadata=recipient_metadata,
            )
            events.extend(result.events)
    iteration: TransitionResult[MediatorTransferState] = TransitionResult(mediator_state, events)
    return iteration


def state_transition(
    mediator_state: Optional[MediatorTransferState],
    state_change: StateChange,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> TransitionResult[Optional[MediatorTransferState]]:
    iteration: TransitionResult[Optional[MediatorTransferState]] = TransitionResult(mediator_state, [])

    if type(state_change) == ActionInitMediator:
        assert isinstance(state_change, ActionInitMediator), MYPY_ANNOTATION
        if mediator_state is None:
            iteration = handle_init(
                state_change=state_change,
                channelidentifiers_to_channels=channelidentifiers_to_channels,
                addresses_to_channel=addresses_to_channel,
                pseudo_random_generator=pseudo_random_generator,
                block_number=block_number,
            )

    elif type(state_change) == Block:
        assert isinstance(state_change, Block), MYPY_ANNOTATION
        assert mediator_state, "Block should be accompanied by a valid mediator state"
        iteration = handle_block(
            mediator_state=mediator_state,
            state_change=state_change,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            addresses_to_channel=addresses_to_channel,
            pseudo_random_generator=pseudo_random_generator,
        )

    elif type(state_change) == ReceiveTransferRefund:
        assert isinstance(state_change, ReceiveTransferRefund), MYPY_ANNOTATION
        assert mediator_state, "ReceiveTransferRefund should be accompanied by a valid mediator state"
        iteration = handle_refundtransfer(
            mediator_state=mediator_state,
            mediator_state_change=state_change,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            addresses_to_channel=addresses_to_channel,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number,
        )

    elif type(state_change) == ReceiveSecretReveal:
        assert isinstance(state_change, ReceiveSecretReveal), MYPY_ANNOTATION
        assert mediator_state, "ReceiveSecretReveal should be accompanied by a valid mediator state"
        iteration = handle_offchain_secretreveal(
            mediator_state=mediator_state,
            mediator_state_change=state_change,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number,
            block_hash=block_hash,
        )

    elif type(state_change) == ContractReceiveSecretReveal:
        assert isinstance(state_change, ContractReceiveSecretReveal), MYPY_ANNOTATION
        assert mediator_state, "ContractReceiveSecretReveal should be accompanied by a valid mediator state"
        iteration = handle_onchain_secretreveal(
            mediator_state=mediator_state,
            onchain_secret_reveal=state_change,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number,
        )

    elif type(state_change) == ReceiveUnlock:
        assert isinstance(state_change, ReceiveUnlock), MYPY_ANNOTATION
        assert mediator_state, "ReceiveUnlock should be accompanied by a valid mediator state"
        iteration = handle_unlock(
            mediator_state=mediator_state,
            state_change=state_change,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
        )

    elif type(state_change) == ReceiveLockExpired:
        assert isinstance(state_change, ReceiveLockExpired), MYPY_ANNOTATION
        assert mediator_state, "ReceiveLockExpired should be accompanied by a valid mediator state"
        iteration = handle_lock_expired(
            mediator_state=mediator_state,
            state_change=state_change,
            channelidentifiers_to_channels=channelidentifiers_to_channels,
            block_number=block_number,
        )

    if iteration.new_state is not None:
        typecheck(iteration.new_state, MediatorTransferState)
        sanity_check(iteration.new_state, channelidentifiers_to_channels)

    return clear_if_finalized(iteration, channelidentifiers_to_channels)