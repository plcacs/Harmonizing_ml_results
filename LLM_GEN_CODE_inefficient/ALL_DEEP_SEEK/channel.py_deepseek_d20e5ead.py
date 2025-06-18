from typing import Any, Dict, List, Optional, Tuple, Union
from eth_utils import encode_hex, keccak, to_hex
from raiden.constants import LOCKSROOT_OF_NO_LOCKS, MAXIMUM_PENDING_TRANSFERS, UINT256_MAX
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, MediationFeeConfig
from raiden.transfer.architecture import Event, StateChange, SuccessOrError, TransitionResult
from raiden.transfer.events import (
    ContractSendChannelBatchUnlock,
    ContractSendChannelClose,
    ContractSendChannelCoopSettle,
    ContractSendChannelSettle,
    ContractSendChannelUpdateTransfer,
    ContractSendChannelWithdraw,
    EventInvalidActionCoopSettle,
    EventInvalidActionSetRevealTimeout,
    EventInvalidActionWithdraw,
    EventInvalidReceivedLockedTransfer,
    EventInvalidReceivedLockExpired,
    EventInvalidReceivedTransferRefund,
    EventInvalidReceivedUnlock,
    EventInvalidReceivedWithdraw,
    EventInvalidReceivedWithdrawExpired,
    EventInvalidReceivedWithdrawRequest,
    SendProcessed,
    SendWithdrawConfirmation,
    SendWithdrawExpired,
    SendWithdrawRequest,
)
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import (
    SendLockedTransfer,
    SendLockExpired,
    SendUnlock,
)
from raiden.transfer.mediated_transfer.mediation_fee import (
    FeeScheduleState,
    calculate_imbalance_fees,
)
from raiden.transfer.mediated_transfer.state import (
    LockedTransferSignedState,
    LockedTransferUnsignedState,
)
from raiden.transfer.mediated_transfer.state_change import (
    ReceiveLockExpired,
    ReceiveTransferRefund,
)
from raiden.transfer.state import (
    CHANNEL_STATES_PRIOR_TO_CLOSED,
    BalanceProofSignedState,
    BalanceProofUnsignedState,
    ChannelState,
    CoopSettleState,
    ExpiredWithdrawState,
    HashTimeLockState,
    NettingChannelEndState,
    NettingChannelState,
    PendingLocksState,
    PendingWithdrawState,
    RouteState,
    TransactionExecutionStatus,
    UnlockPartialProofState,
    get_address_metadata,
    message_identifier_from_prng,
)
from raiden.transfer.state_change import (
    ActionChannelClose,
    ActionChannelCoopSettle,
    ActionChannelSetRevealTimeout,
    ActionChannelWithdraw,
    Block,
    ContractReceiveChannelBatchUnlock,
    ContractReceiveChannelClosed,
    ContractReceiveChannelDeposit,
    ContractReceiveChannelSettled,
    ContractReceiveChannelWithdraw,
    ContractReceiveUpdateTransfer,
    ReceiveUnlock,
    ReceiveWithdrawConfirmation,
    ReceiveWithdrawExpired,
    ReceiveWithdrawRequest,
)
from raiden.transfer.utils import hash_balance_data
from raiden.utils.formatting import to_checksum_address
from raiden.utils.packing import pack_balance_proof, pack_withdraw
from raiden.utils.signer import recover
from raiden.utils.typing import (
    MYPY_ANNOTATION,
    Address,
    AddressMetadata,
    Any,
    Balance,
    BlockExpiration,
    BlockHash,
    BlockNumber,
    BlockTimeout,
    ChainID,
    ChannelID,
    Dict,
    EncodedData,
    InitiatorAddress,
    List,
    LockedAmount,
    Locksroot,
    LockType,
    MessageID,
    NamedTuple,
    Nonce,
    Optional,
    PaymentAmount,
    PaymentID,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    Signature,
    TargetAddress,
    TokenAmount,
    TokenNetworkAddress,
    Tuple,
    Union,
    WithdrawAmount,
    typecheck,
)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from raiden.raiden_service import RaidenService  # noqa: F401

# This should be changed to `Union[str, PendingLocksState]`
PendingLocksStateOrError = Tuple[bool, Optional[str], Optional[PendingLocksState]]
EventsOrError = Tuple[bool, List[Event], Optional[str]]
BalanceProofData = Tuple[Locksroot, Nonce, TokenAmount, LockedAmount]
SendUnlockAndPendingLocksState = Tuple[SendUnlock, PendingLocksState]


class UnlockGain(NamedTuple):
    from_our_locks: TokenAmount
    from_partner_locks: TokenAmount


class ChannelUsability(Enum):
    USABLE = True
    NOT_OPENED = "channel is not open"
    INVALID_SETTLE_TIMEOUT = "channel settle timeout is too low"
    CHANNEL_REACHED_PENDING_LIMIT = "channel reached limit of pending transfers"
    CHANNEL_DOESNT_HAVE_ENOUGH_DISTRIBUTABLE = "channel doesn't have enough distributable tokens"
    CHANNEL_BALANCE_PROOF_WOULD_OVERFLOW = "channel balance proof would overflow"
    LOCKTIMEOUT_MISMATCH = "the lock timeout can not be used with the channel"


def get_safe_initial_expiration(
    block_number: BlockNumber, reveal_timeout: BlockTimeout, lock_timeout: BlockTimeout = None
) -> BlockExpiration:
    """Returns the upper bound block expiration number used by the initiator
    of a transfer or a withdraw.

    The `reveal_timeout` defines how many blocks it takes for a transaction to
    be mined under congestion. The expiration is defined in terms of
    `reveal_timeout`.

    It must be at least `reveal_timeout` to allow a lock or withdraw to be used
    on-chain under congestion. Ideally it should not be larger than `2 *
    reveal_timeout`, otherwise for off-chain transfers Raiden would be slower
    than blockchain.
    """
    if lock_timeout:
        expiration = block_number + lock_timeout
    else:
        expiration = block_number + reveal_timeout * 2

    # Other nodes may not see the same block we do, so allow for a difference
    # of 1 block. A mediator node can fail to use an open channel if the lock
    # timeout is greater than the settle timeout due to different blocks being
    # seen between nodes. This delays mediation for at least one block time
    # and therefore increases payment time.
    return BlockExpiration(expiration - 1)


def get_sender_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    """Compute the block at which an expiration message can be sent without
    worrying about blocking the message queue.

    The expiry messages will be rejected if the expiration block has not been
    confirmed. Additionally the sender can account for possible delays in the
    receiver, so a few additional blocks are used to avoid hanging the channel.
    """
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS * 2)


def get_receiver_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    """Returns the block number at which the receiver can accept an expiry
    message.

    The receiver must wait for the block at which the expired message to be
    confirmed. This is necessary to handle reorgs, e.g. which could hide a
    secret registration.
    """
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS)


def is_channel_usable_for_mediation(
    channel_state: NettingChannelState,
    transfer_amount: PaymentWithFeeAmount,
    lock_timeout: BlockTimeout,
) -> bool:
    """True if the channel can safely used to mediate a transfer for the given
    parameters.

    This will make sure that:

    - The channel has capacity.
    - The number of locks can be claimed on-chain.
    - The transfer amount does not overflow.
    - The lock expiration is smaller than the settlement window.

    The number of locks has to be checked because the gas usage will increase
    linearly with the number of locks in it, this has to be limited to a value
    lower than the block gas limit constraints.

    The lock expiration has to be smaller than the channel's settlement window
    because otherwise it is possible to employ attacks, where an attacker opens
    two channels to the victim, with different settlement windows. The channel
    with lower settlement is used to start a payment to the other channel, if
    the lock's expiration is allowed to be larger than the settlement window,
    then the attacker can close and settle the incoming channel before the lock
    expires, and claim the lock on the outgoing channel by registering the
    secret on-chain.
    """

    channel_usable = is_channel_usable_for_new_transfer(
        channel_state, transfer_amount, lock_timeout
    )

    return channel_usable is ChannelUsability.USABLE


def is_channel_usable_for_new_transfer(
    channel_state: NettingChannelState,
    transfer_amount: PaymentWithFeeAmount,
    lock_timeout: Optional[BlockTimeout],
) -> ChannelUsability:
    """True if the channel can be used to start a new transfer.

    This will make sure that:

    - The channel has capacity.
    - The number of locks can be claimed on-chain.
    - The transfer amount does not overflow.
    - The settlement window is large enough to allow the secret to be
      registered on-chain.
    - lock_timeout, if provided, is within allowed range (reveal_timeout, settle_timeout]

    The number of locks has to be checked because the gas usage will increase
    linearly with the number of locks in it, this has to be limited to a value
    lower than the block gas limit constraints.
    """
    pending_transfers = get_number_of_pending_transfers(channel_state.our_state)
    distributable = get_distributable(channel_state.our_state, channel_state.partner_state)

    lock_timeout_valid = lock_timeout is None or (
        lock_timeout <= channel_state.settle_timeout
        and lock_timeout > channel_state.reveal_timeout
    )

    # The settle_timeout can be chosen independently by our partner. That means
    # it is possible for a malicious partner to choose a settlement timeout so
    # small that it is not possible to register the secret on-chain.
    #
    # This is true for version 0.25.0 of the smart contracts, since there is
    # nothing the client can do to prevent the channel from being open the only
    # option is to ignore the channel.
    is_valid_settle_timeout = channel_state.settle_timeout >= channel_state.reveal_timeout * 2

    if get_status(channel_state) != ChannelState.STATE_OPENED:
        return ChannelUsability.NOT_OPENED

    if not is_valid_settle_timeout:
        return ChannelUsability.INVALID_SETTLE_TIMEOUT

    if pending_transfers >= MAXIMUM_PENDING_TRANSFERS:
        return ChannelUsability.CHANNEL_REACHED_PENDING_LIMIT

    if transfer_amount > distributable:
        return ChannelUsability.CHANNEL_DOESNT_HAVE_ENOUGH_DISTRIBUTABLE

    if not is_valid_amount(channel_state.our_state, transfer_amount):
        return ChannelUsability.CHANNEL_BALANCE_PROOF_WOULD_OVERFLOW

    if not lock_timeout_valid:
        return ChannelUsability.LOCKTIMEOUT_MISMATCH

    return ChannelUsability.USABLE


def is_lock_pending(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    """True if the `secrethash` corresponds to a lock that is pending to be claimed
    and didn't expire.
    """
    return (
        secrethash in end_state.secrethashes_to_lockedlocks
        or secrethash in end_state.secrethashes_to_unlockedlocks
        or secrethash in end_state.secrethashes_to_onchain_unlockedlocks
    )


def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    """True if the `secrethash` is for a lock with an unknown secret."""
    return secrethash in end_state.secrethashes_to_lockedlocks


def is_lock_expired(
    end_state: NettingChannelEndState,
    lock: LockType,
    block_number: BlockNumber,
    lock_expiration_threshold: BlockExpiration,
) -> SuccessOrError:
    """Determine whether a lock has expired.

    The lock has expired if both:

        - The secret was not registered on-chain in time.
        - The current block exceeds lock's expiration + confirmation blocks.
    """

    secret_registered_on_chain = lock.secrethash in end_state.secrethashes_to_onchain_unlockedlocks
    if secret_registered_on_chain:
        return SuccessOrError("lock has been unlocked on-chain")

    if block_number < lock_expiration_threshold:
        return SuccessOrError(
            f"current block number ({block_number}) is not larger than "
            f"lock.expiration + confirmation blocks ({lock_expiration_threshold})"
        )
    return SuccessOrError()


def is_transfer_expired(
    transfer: LockedTransferSignedState,
    affected_channel: NettingChannelState,
    block_number: BlockNumber,
) -> bool:
    lock_expiration_threshold = get_sender_expiration_threshold(transfer.lock.expiration)
    has_lock_expired = is_lock_expired(
        end_state=affected_channel.our_state,
        lock=transfer.lock,
        block_number=block_number,
        lock_expiration_threshold=lock_expiration_threshold,
    )
    return has_lock_expired.ok


def is_withdraw_expired(block_number: BlockNumber, expiration_threshold: BlockExpiration) -> bool:
    """Determine whether a withdraw has expired.

    The withdraw has expired if the current block exceeds
    the withdraw's expiration + confirmation blocks.
    """
    return block_number >= expiration_threshold


def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    """True if the `secrethash` is for a lock with a known secret."""
    return (
        secrethash in end_state.secrethashes_to_unlockedlocks
        or secrethash in end_state.secrethashes_to_onchain_unlockedlocks
    )


def is_secret_known_offchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    """True if the `secrethash` is for a lock with a known secret."""
    return secrethash in end_state.secrethashes_to_unlockedlocks


def is_secret_known_onchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    """True if the `secrethash` is for a lock with a known secret."""
    return secrethash in end_state.secrethashes_to_onchain_unlockedlocks


def is_valid_channel_total_withdraw(channel_total_withdraw: TokenAmount) -> bool:
    """Sanity check for the channel's total withdraw.

    The channel's total deposit is:

        p1.total_deposit + p2.total_deposit

    The channel's total withdraw is:

        p1.total_withdraw + p2.total_withdraw

    The smart contract forces:

        - The channel's total deposit to fit in a UINT256.
        - The channel's withdraw must be in the range [0,channel_total_deposit].

    Because the `total_withdraw` must be in the range [0,channel_deposit], and
    the maximum value for channel_deposit is UINT256, the overflow below must
    never happen, otherwise there is a smart contract bug.
    """
    return channel_total_withdraw <= UINT256_MAX


def is_valid_withdraw(
    withdraw_request: Union[ReceiveWithdrawRequest, ReceiveWithdrawConfirmation]
) -> SuccessOrError:
    """True if the signature of the message corresponds is valid.

    This predicate is intentionally only checking the signature against the
    message data, and not the expected data. Before this check the fields of
    the message must be validated.
    """
    packed = pack_withdraw(
        canonical_identifier=withdraw_request.canonical_identifier,
        participant=withdraw_request.participant,
        total_withdraw=withdraw_request.total_withdraw,
        expiration_block=withdraw_request.expiration,
    )

    return is_valid_signature(
        data=packed, signature=withdraw_request.signature, sender_address=withdraw_request.sender
    )


def get_secret(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[Secret]:
    """Returns `secret` if the `secrethash` is for a lock with a known secret."""
    partial_unlock_proof = end_state.secrethashes_to_unlockedlocks.get(secrethash)

    if partial_unlock_proof is None:
        partial_unlock_proof = end_state.secrethashes_to_onchain_unlockedlocks.get(secrethash)

    if partial_unlock_proof is not None:
        return partial_unlock_proof.secret

    return None


def is_balance_proof_safe_for_onchain_operations(
    balance_proof: BalanceProofSignedState,
) -> bool:
    """Check if the balance proof would overflow onchain."""
    total_amount = balance_proof.transferred_amount + balance_proof.locked_amount
    return total_amount <= UINT256_MAX


def is_valid_amount(
    end_state: NettingChannelEndState,
    amount: Union[TokenAmount, PaymentAmount, PaymentWithFeeAmount],
) -> bool:
    (_, _, current_transferred_amount, current_locked_amount) = get_current_balanceproof(end_state)

    transferred_amount_after_unlock = current_transferred_amount + current_locked_amount + amount

    return transferred_amount_after_unlock <= UINT256_MAX


def is_valid_signature(
    data: bytes, signature: Signature, sender_address: Address
) -> SuccessOrError:
    try:
        signer_address = recover(data=data, signature=signature)
        # InvalidSignature is raised by raiden.utils.signer.recover if signature
        # is not bytes or has the incorrect length.
        # ValueError is raised if the PublicKey instantiation failed.
        # Exception is raised if the public key recovery failed.
    except Exception:  # pylint: disable=broad-except
        return SuccessOrError("Signature invalid, could not be recovered.")

    is_correct_sender = sender_address == signer_address
    if is_correct_sender:
        return SuccessOrError()

    return SuccessOrError("Signature was valid but the expected address does not match.")


def is_valid_balanceproof_signature(
    balance_proof: BalanceProofSignedState, sender_address: Address
) -> SuccessOrError:
    balance_hash = hash_balance_data(
        balance_proof.transferred_amount, balance_proof.locked_amount, balance_proof.locksroot
    )

    # The balance proof must be tied to a single channel instance, through the
    # chain_id, token_network_address, and channel_identifier, otherwise the
    # on-chain contract would be susceptible to replay attacks across channels.
    #
    # The balance proof must also authenticate the offchain balance (blinded in
    # the balance_hash field), and authenticate the rest of message data
    # (blinded in additional_hash).
    data_that_was_signed = pack_balance_proof(
        nonce=balance_proof.nonce,
        balance_hash=balance_hash,
        additional_hash=balance_proof.message_hash,
        canonical_identifier=CanonicalIdentifier(
            chain_identifier=balance_proof.chain_id,
            token_network_address=balance_proof.token_network_address,
            channel_identifier=balance_proof.channel_identifier,
        ),
    )

    return is_valid_signature(
        data=data_that_was_signed, signature=balance_proof.signature, sender_address=sender_address
    )


def is_balance_proof_usable_onchain(
    received_balance_proof: BalanceProofSignedState,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
) -> SuccessOrError:
    """Checks the balance proof can be used on-chain.

    For a balance proof to be valid it must be newer than the previous one,
    i.e. the nonce must increase, the signature must tie the balance proof to
    the correct channel, and the values must not result in an under/overflow
    onchain.

    Important: This predicate does not validate all the message fields. The
    fields locksroot, transferred_amount, and locked_amount **MUST** be
    validated elsewhere based on the message type.
    """
    expected_nonce = get_next_nonce(sender_state)

    is_valid_signature = is_valid_balanceproof_signature(
        received_balance_proof, sender_state.address
    )

    # TODO: Accept unlock messages if the node has not yet sent a transaction
    # with the balance proof to the blockchain, this will save one call to
    # unlock on-chain for the non-closing party.
    if get_status(channel_state) != ChannelState.STATE_OPENED:
        # The channel must be opened, otherwise if receiver is the closer, the
        # balance proof cannot be used onchain.
        return SuccessOrError("The channel is already closed.")

    elif received_balance_proof.channel_identifier != channel_state.identifier:
        # Informational message, the channel_identifier **validated by the
        # signature** must match for the balance_proof to be valid.
        return SuccessOrError(
            f"channel_identifier does not match. "
            f"expected: {channel_state.identifier} "
            f"got: {received_balance_proof.channel_identifier}."
        )

    elif received_balance_proof.token_network_address != channel_state.token_network_address:
        # Informational message, the token_network_address **validated by
        # the signature** must match for the balance_proof to be valid.
        return SuccessOrError(
            f"token_network_address does not match. "
            f"expected: {to_checksum_address(channel_state.token_network_address)} "
            f"got: {to_checksum_address(received_balance_proof.token_network_address)}."
        )

    elif received_balance_proof.chain_id != channel_state.chain_id:
        # Informational message, the chain_id **validated by the signature**
        # must match for the balance_proof to be valid.
        return SuccessOrError(
            f"chain_id does not match channel's "
            f"chain_id. expected: {channel_state.chain_id} "
            f"got: {received_balance_proof.chain_id}."
        )

    elif not is_balance_proof_safe_for_onchain_operations(received_balance_proof):
        transferred_amount_after_unlock = (
            received_balance_proof.transferred_amount + received_balance_proof.locked_amount
        )
        return SuccessOrError(
            f"Balance proof total transferred amount would overflow onchain. "
            f"max: {UINT256_MAX} result would be: {transferred_amount_after_unlock}"
        )

    elif received_balance_proof.nonce != expected_nonce:
        # The nonces must increase sequentially, otherwise there is a
        # synchronization problem.
        return SuccessOrError(
            f"Nonce did not change sequentially, expected: {expected_nonce} "
            f"got: {received_balance_proof.nonce}."
        )

    else:
        # The signature must be valid, otherwise the balance proof cannot be
        # used onchain.
        return is_valid_signature


def is_valid_lockedtransfer(
    transfer_state: LockedTransferSignedState,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
) -> PendingLocksStateOrError:
    return valid_lockedtransfer_check(
        channel_state=channel_state,
        sender_state=sender_state,
        receiver_state=receiver_state,
        message_name="LockedTransfer",
        received_balance_proof=transfer_state.balance_proof,
        lock=transfer_state.lock,
    )


def is_valid_lock_expired(
    state_change: ReceiveLockExpired,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
    block_number: BlockNumber,
) -> PendingLocksStateOrError:
    secrethash = state_change.secrethash
    received_balance_proof = state_change.balance_proof

    # If the lock was not found in locked locks, this means that we've received
    # the secret for the locked transfer but we haven't unlocked it yet. Lock
    # expiry in this case could still happen which means that we have to make
    # sure that we check for "unclaimed" locks in our check.
    lock = channel_state.partner_state.secrethashes_to_lockedlocks.get(secrethash)
    if not lock:
        partial_lock = channel_state.partner_state.secrethashes_to_unlockedlocks.get(secrethash)
        if partial_lock:
            lock = partial_lock.lock

    secret_registered_on_chain = (
        secrethash in channel_state.partner_state.secrethashes_to_onchain_unlockedlocks
    )

    current_balance_proof = get_current_balanceproof(sender_state)
    _, _, current_transferred_amount, current_locked_amount = current_balance_proof

    is_valid_balance_proof = is_balance_proof_usable_onchain(
        received_balance_proof=received_balance_proof,
        channel_state=channel_state,
        sender_state=sender_state,
    )

    if lock:
        pending_locks = compute_locks_without(
            sender_state.pending_locks, EncodedData(bytes(lock.encoded))
        expected_locked_amount = current_locked_amount - lock.amount

    result: PendingLocksStateOrError = (False, None, None)

    if secret_registered_on_chain:
        msg = "Invalid LockExpired message. Lock was unlocked on-chain."
        result = (False, msg, None)

    elif lock is None:
        msg = (
            f"Invalid LockExpired message. "
            f"Lock with secrethash {to_hex(secrethash)} is not known."
        )
        result = (False, msg, None)

    elif not is_valid_balance_proof:
        msg = f"Invalid LockExpired message. {is_valid_balance_proof.as_error_message}"
        result = (False, msg, None)

    elif pending_locks is None:
        msg = "Invalid LockExpired message. Same lock handled twice."
        result = (False, msg, None)

    else:
        locksroot_without_lock = compute_locksroot(pending_locks)
        check_lock_expired = is_lock_expired(
            end_state=receiver_state,
            lock=lock,
            block_number=block_number,
            lock_expiration_threshold=get_receiver_expiration_threshold(lock.expiration),
        )

        if not check_lock_expired:
            msg = f"Invalid LockExpired message. {check_lock_expired.as_error_message}"
            result = (False, msg, None)

        elif received_balance_proof.locksroot != locksroot_without_lock:
            # The locksroot must be updated, and the expired lock must be *removed*
            msg = (
                "Invalid LockExpired message. "
                "Balance proof's locksroot didn't match, expected: {} got: {}."
            ).format(
                encode_hex(locksroot_without_lock), encode_hex(received_balance_proof.locksroot)
            )

            result = (False, msg, None)

        elif received_balance_proof.transferred_amount != current_transferred_amount:
            # Given an expired lock, transferred amount should stay the same
            msg = (
                "Invalid LockExpired message. "
                "Balance proof's transferred_amount changed, expected: {} got: {}."
            ).format(current_transferred_amount, received_balance_proof.transferred_amount)

            result = (False, msg, None)

        elif received_balance_proof.locked_amount != expected_locked_amount:
            # locked amount should be the same found inside the balance proof
            msg = (
                "Invalid LockExpired message. "
                "Balance proof's locked_amount is invalid, expected: {} got: {}."
            ).format(expected_locked_amount, received_balance_proof.locked_amount)

            result = (False, msg, None)

        else:
            result = (True, None, pending_locks)

    return result


def valid_lockedtransfer_check(
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
    message_name: str,
    received_balance_proof: BalanceProofSignedState,
    lock: HashTimeLockState,
) -> PendingLocksStateOrError:

    current_balance_proof = get_current_balanceproof(sender_state)
    pending_locks = compute_locks_with(sender_state.pending_locks, lock)

    _, _, current_transferred_amount, current_locked_amount = current_balance_proof
    distributable = get_distributable(sender_state, receiver_state)
    expected_locked_amount = current_locked_amount + lock.amount

    is_valid_balance_proof = is_balance_proof_usable_onchain(
        received_balance_proof=received_balance_proof,
        channel_state=channel_state,
        sender_state=sender_state,
    )

    result: PendingLocksStateOrError = (False, None, None)

    if not is_valid_balance_proof:
        msg = f"Invalid {message_name} message. {is_valid_balance_proof.as_error_message}"
        result = (False, msg, None)

    elif pending_locks is None:
        msg = f"Invalid {message_name} message. Same lock handled twice."
        result = (False, msg, None)

    elif len(pending_locks.locks) > MAXIMUM_PENDING_TRANSFERS:
        msg = (
            f"Invalid {message_name} message. Adding the transfer would exceed the allowed "
            f"limit of {MAXIMUM_PENDING_TRANSFERS} pending transfers per channel."
        )
        result = (False, msg, None)

    else:
        locksroot_with_lock = compute_locksroot(pending_locks)

        if received_balance_proof.locksroot != locksroot_with_lock:
            # The locksroot must be updated to include the new lock
            msg = (
                "Invalid {} message. "
                "Balance proof's locksroot didn't match, expected: {} got: {}."
            ).format(
                message_name,
                encode_hex(locksroot_with_lock),
                encode_hex(received_balance_proof.locksroot),
            )

            result = (False, msg, None)

        elif received_balance_proof.transferred_amount != current_transferred_amount:
            # Mediated transfers must not change transferred_amount
            msg = (
                "Invalid {} message. "
                "Balance proof's transferred_amount changed, expected: {} got: {}."
            ).format(
                message_name, current_transferred_amount, received_balance_proof.transferred_amount
            )

            result = (False, msg, None)

        elif received_balance_proof.locked_amount != expected_locked_amount:
            # Mediated transfers must increase the locked_amount by lock.amount
            msg = (
                "Invalid {} message. "
                "Balance proof's locked_amount is invalid, expected: {} got: {}."
            ).format(message_name, expected_locked_amount, received_balance_proof.locked_amount)

            result = (False, msg, None)

        # the locked amount is limited to the current available balance, otherwise
        # the sender is attempting to game the protocol and do a double spend
        elif lock.amount > distributable:
            msg = (
                "Invalid {} message. "
                "Lock amount larger than the available distributable, "
                "lock amount: {} maximum distributable: {}"
            ).format(message_name, lock.amount, distributable)

            result = (False, msg, None)

        else:
            result = (True, None, pending_locks)

    return result


def refund_transfer_matches_transfer(
    refund_transfer: LockedTransferSignedState, transfer: LockedTransferUnsignedState
) -> bool:
    refund_transfer_sender = refund_transfer.balance_proof.sender
    # Ignore a refund from the target
    if TargetAddress(refund_transfer_sender) == transfer.target:
        return False

    return (
        transfer.payment_identifier == refund_transfer.payment_identifier
        and transfer.lock.amount == refund_transfer.lock.amount
        and transfer.lock.secrethash == refund_transfer.lock.secrethash
        and transfer.target == refund_transfer.target
        and transfer.lock.expiration == refund_transfer.lock.expiration
        and
        # The refund transfer is not tied to the other direction of the same
        # channel, it may reach this node through a different route depending
        # on the path finding strategy
        # original_receiver == refund_transfer_sender and
        transfer.token == refund_transfer.token
    )


def is_valid_refund(
    refund: ReceiveTransferRefund,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
    received_transfer: LockedTransferUnsignedState,
) -> PendingLocksStateOrError:
    is_valid_locked_transfer, msg, pending_locks = valid_lockedtransfer_check(
        channel_state,
        sender_state,
        receiver_state,
        "RefundTransfer",
        refund.transfer.balance_proof,
        refund.transfer.lock,
    )

    if not is_valid_locked_transfer:
        return False, msg, None

    if not refund_transfer_matches_transfer(refund.transfer, received_transfer):
        return False, "Refund transfer did not match the received transfer", None

    return True, "", pending_locks


def is_valid_unlock(
    unlock: ReceiveUnlock, channel_state: NettingChannelState, sender_state: NettingChannelEndState
) -> PendingLocksStateOrError:
    received_balance_proof = unlock.balance_proof
    current_balance_proof = get_current_balanceproof(sender_state)

    lock = get_lock(sender_state, unlock.secrethash)

    if lock is None:
        msg = "Invalid Unlock message. There is no corresponding lock for {}".format(
            encode_hex(unlock.secrethash)
        )
        return (False, msg, None)

    pending_locks = compute_locks_without(
        sender_state.pending_locks, EncodedData(bytes(lock.encoded))
    )

    if pending_locks is None:
        msg = f"Invalid unlock message. The lock is unknown {encode_hex(lock.encoded)}"
        return (False, msg, None)

    locksroot_without_lock = compute_locksroot(pending_locks)

    _, _, current_transferred_amount, current_locked_amount = current_balance_proof

    expected_transferred_amount = current_transferred_amount + TokenAmount(lock.amount)
    expected_locked_amount = current_locked_amount - lock.amount

    is_valid_balance_proof = is_balance_proof_usable_onchain(
        received_balance_proof=received_balance_proof,
        channel_state=channel_state,
        sender_state=sender_state,
    )

    result: PendingLocksStateOrError = (False, None, None)

    if not is_valid_balance_proof:
        msg = f"Invalid Unlock message. {is_valid_balance_proof.as_error_message}"
        result = (False, msg, None)

    elif received_balance_proof.locksroot != locksroot_without_lock:
        # Unlock messages remove a known lock, the new locksroot must have only
        # that lock removed, otherwise the sender may be trying to remove
        # additional locks.
        msg = (
            "Invalid Unlock message. "
            "Balance proof's locksroot didn't match, expected: {} got: {}."
        ).format(encode_hex(locksroot_without_lock), encode_hex(received_balance_proof.locksroot))

       