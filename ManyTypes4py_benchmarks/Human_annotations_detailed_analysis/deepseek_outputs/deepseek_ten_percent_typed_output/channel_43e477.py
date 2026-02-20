# pylint: disable=too-many-lines,unused-argument
import random
from enum import Enum
from functools import singledispatch
from typing import TYPE_CHECKING

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
    block_number: BlockNumber,
    reveal_timeout: BlockTimeout,
    lock_timeout: Optional[BlockTimeout] = None,
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
    transfer_amount: PaymentAmount,
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
    transfer_amount: PaymentAmount,
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
    lock: HashTimeLockState,
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
    withdraw_request: ReceiveWithdrawRequest,
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
    amount: PaymentAmount,
) -> bool:
    (_, _, current_transferred_amount, current_locked_amount) = get_current_balanceproof(end_state)

    transferred_amount_after_unlock = current_transferred_amount + current_locked_amount + amount

    return transferred_amount_after_unlock <= UINT256_MAX


def is_valid_signature(
    data: bytes,
    signature: Signature,
    sender_address: Address,
) -> SuccessOrError:
    try:
        signer_address = recover(data=data, signature=signature)
        # InvalidSignature is raised by raiden.utils.signer.recover if signature
        # is not bytes or has the incorrect length.
        # ValueError is raised if the PublicKey instantiation failed.
        # Exception is raised if the public key recovery failed.
    except Exception:  # pylint: disable=broad-except
        return SuccessOrError("Signature invalid, could not be recovered.")

    is_correct_sender = sender_address