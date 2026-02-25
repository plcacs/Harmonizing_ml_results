# pylint: disable=too-many-lines,unused-argument
import random
from enum import Enum
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union, TypeVar, Generic, cast

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
    Balance,
    BlockExpiration,
    BlockHash,
    BlockNumber,
    BlockTimeout,
    ChainID,
    ChannelID,
    EncodedData,
    InitiatorAddress,
    LockedAmount,
    Locksroot,
    LockType,
    MessageID,
    Nonce,
    PaymentAmount,
    PaymentID,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    Signature,
    TargetAddress,
    TokenAmount,
    TokenNetworkAddress,
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

T = TypeVar('T')

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
    lock_timeout: Optional[BlockTimeout] = None
) -> BlockExpiration:
    if lock_timeout:
        expiration = block_number + lock_timeout
    else:
        expiration = block_number + reveal_timeout * 2
    return BlockExpiration(expiration - 1)


def get_sender_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS * 2)


def get_receiver_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS)


def is_channel_usable_for_mediation(
    channel_state: NettingChannelState,
    transfer_amount: PaymentWithFeeAmount,
    lock_timeout: BlockTimeout,
) -> bool:
    channel_usable = is_channel_usable_for_new_transfer(
        channel_state, transfer_amount, lock_timeout
    )
    return channel_usable is ChannelUsability.USABLE


def is_channel_usable_for_new_transfer(
    channel_state: NettingChannelState,
    transfer_amount: PaymentWithFeeAmount,
    lock_timeout: Optional[BlockTimeout],
) -> ChannelUsability:
    pending_transfers = get_number_of_pending_transfers(channel_state.our_state)
    distributable = get_distributable(channel_state.our_state, channel_state.partner_state)

    lock_timeout_valid = lock_timeout is None or (
        lock_timeout <= channel_state.settle_timeout
        and lock_timeout > channel_state.reveal_timeout
    )

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
    return (
        secrethash in end_state.secrethashes_to_lockedlocks
        or secrethash in end_state.secrethashes_to_unlockedlocks
        or secrethash in end_state.secrethashes_to_onchain_unlockedlocks
    )


def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_lockedlocks


def is_lock_expired(
    end_state: NettingChannelEndState,
    lock: LockType,
    block_number: BlockNumber,
    lock_expiration_threshold: BlockExpiration,
) -> SuccessOrError:
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
    return block_number >= expiration_threshold


def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return (
        secrethash in end_state.secrethashes_to_unlockedlocks
        or secrethash in end_state.secrethashes_to_onchain_unlockedlocks
    )


def is_secret_known_offchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_unlockedlocks


def is_secret_known_onchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_onchain_unlockedlocks


def is_valid_channel_total_withdraw(channel_total_withdraw: TokenAmount) -> bool:
    return channel_total_withdraw <= UINT256_MAX


def is_valid_withdraw(
    withdraw_request: Union[ReceiveWithdrawRequest, ReceiveWithdrawConfirmation]
) -> SuccessOrError:
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
    partial_unlock_proof = end_state.secrethashes_to_unlockedlocks.get(secrethash)

    if partial_unlock_proof is None:
        partial_unlock_proof = end_state.secrethashes_to_onchain_unlockedlocks.get(secrethash)

    if partial_unlock_proof is not None:
        return partial_unlock_proof.secret

    return None


def is_balance_proof_safe_for_onchain_operations(
    balance_proof: BalanceProofSignedState,
) -> bool:
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
    data: bytes, 
    signature: Signature, 
    sender_address: Address
) -> SuccessOrError:
    try:
        signer_address = recover(data=data, signature=signature)
    except Exception:  # pylint: disable=broad-except
        return SuccessOrError("Signature invalid, could not be recovered.")

    is_correct_sender = sender_address == signer_address
    if is_correct_sender:
        return SuccessOrError()
    return SuccessOrError("Signature was valid but the expected address does not match.")


def is_valid_balanceproof_signature(
    balance_proof: BalanceProofSignedState, 
    sender_address: Address
) -> SuccessOrError:
    balance_hash = hash_balance_data(
        balance_proof.transferred_amount, 
        balance_proof.locked_amount, 
        balance_proof.locksroot
    )

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
        data=data_that_was_signed, 
        signature=balance_proof.signature, 
        sender_address=sender_address
    )


def is_balance_proof_usable_onchain(
    received_balance_proof: BalanceProofSignedState,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
) -> SuccessOrError:
    expected_nonce = get_next_nonce(sender_state)
    is_valid_signature = is_valid_balanceproof_signature(received_balance_proof, sender_state.address)

    if get_status(channel_state) != ChannelState.STATE_OPENED:
        return SuccessOrError("The channel is already closed.")
    elif received_balance_proof.channel_identifier != channel_state.identifier:
        return SuccessOrError(
            f"channel_identifier does not match. "
            f"expected: {channel_state.identifier} "
            f"got: {received_balance_proof.channel_identifier}."
        )
    elif received_balance_proof.token_network_address != channel_state.token_network_address:
        return SuccessOrError(
            f"token_network_address does not match. "
            f"expected: {to_checksum_address(channel_state.token_network_address)} "
            f"got: {to_checksum_address(received_balance_proof.token_network_address)}."
        )
    elif received_balance_proof.chain_id != channel_state.chain_id:
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
        return SuccessOrError(
            f"Nonce did not change sequentially, expected: {expected_nonce} "
            f"got: {received_balance_proof.nonce}."
        )
    else:
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
        )
        expected_locked_amount = current_locked_amount - lock.amount

    result: PendingLocksStateOrError = (False, None, None)

    if secret_registered_on_chain:
        msg = "Invalid LockExpired message. Lock was unlocked on-chain."
        result = (False, msg, None)
    elif lock is None:
        msg =