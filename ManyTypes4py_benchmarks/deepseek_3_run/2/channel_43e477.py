import random
from enum import Enum
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union
from eth_utils import encode_hex, keccak, to_hex
from raiden.constants import LOCKSROOT_OF_NO_LOCKS, MAXIMUM_PENDING_TRANSFERS, UINT256_MAX
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, MediationFeeConfig
from raiden.transfer.architecture import Event, StateChange, SuccessOrError, TransitionResult
from raiden.transfer.events import ContractSendChannelBatchUnlock, ContractSendChannelClose, ContractSendChannelCoopSettle, ContractSendChannelSettle, ContractSendChannelUpdateTransfer, ContractSendChannelWithdraw, EventInvalidActionCoopSettle, EventInvalidActionSetRevealTimeout, EventInvalidActionWithdraw, EventInvalidReceivedLockedTransfer, EventInvalidReceivedLockExpired, EventInvalidReceivedTransferRefund, EventInvalidReceivedUnlock, EventInvalidReceivedWithdraw, EventInvalidReceivedWithdrawExpired, EventInvalidReceivedWithdrawRequest, SendProcessed, SendWithdrawConfirmation, SendWithdrawExpired, SendWithdrawRequest
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import SendLockedTransfer, SendLockExpired, SendUnlock
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState, calculate_imbalance_fees
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState, LockedTransferUnsignedState
from raiden.transfer.mediated_transfer.state_change import ReceiveLockExpired, ReceiveTransferRefund
from raiden.transfer.state import CHANNEL_STATES_PRIOR_TO_CLOSED, BalanceProofSignedState, BalanceProofUnsignedState, ChannelState, CoopSettleState, ExpiredWithdrawState, HashTimeLockState, NettingChannelEndState, NettingChannelState, PendingLocksState, PendingWithdrawState, RouteState, TransactionExecutionStatus, UnlockPartialProofState, get_address_metadata, message_identifier_from_prng
from raiden.transfer.state_change import ActionChannelClose, ActionChannelCoopSettle, ActionChannelSetRevealTimeout, ActionChannelWithdraw, Block, ContractReceiveChannelBatchUnlock, ContractReceiveChannelClosed, ContractReceiveChannelDeposit, ContractReceiveChannelSettled, ContractReceiveChannelWithdraw, ContractReceiveUpdateTransfer, ReceiveUnlock, ReceiveWithdrawConfirmation, ReceiveWithdrawExpired, ReceiveWithdrawRequest
from raiden.transfer.utils import hash_balance_data
from raiden.utils.formatting import to_checksum_address
from raiden.utils.packing import pack_balance_proof, pack_withdraw
from raiden.utils.signer import recover
from raiden.utils.typing import MYPY_ANNOTATION, Address, AddressMetadata, Balance, BlockExpiration, BlockHash, BlockNumber, BlockTimeout, ChainID, ChannelID, EncodedData, InitiatorAddress, LockedAmount, Locksroot, LockType, MessageID, Nonce, PaymentAmount, PaymentID, PaymentWithFeeAmount, Secret, SecretHash, Signature, TargetAddress, TokenAmount, TokenNetworkAddress, WithdrawAmount, typecheck
if TYPE_CHECKING:
    from raiden.raiden_service import RaidenService
PendingLocksStateOrError = Tuple[bool, Optional[str], Optional[PendingLocksState]]
EventsOrError = Tuple[bool, List[Event], Optional[str]]
BalanceProofData = Tuple[Locksroot, Nonce, TokenAmount, LockedAmount]
SendUnlockAndPendingLocksState = Tuple[SendUnlock, PendingLocksState]

class UnlockGain(NamedTuple):
    pass

class ChannelUsability(Enum):
    USABLE = True
    NOT_OPENED = 'channel is not open'
    INVALID_SETTLE_TIMEOUT = 'channel settle timeout is too low'
    CHANNEL_REACHED_PENDING_LIMIT = 'channel reached limit of pending transfers'
    CHANNEL_DOESNT_HAVE_ENOUGH_DISTRIBUTABLE = "channel doesn't have enough distributable tokens"
    CHANNEL_BALANCE_PROOF_WOULD_OVERFLOW = 'channel balance proof would overflow'
    LOCKTIMEOUT_MISMATCH = 'the lock timeout can not be used with the channel'

def get_safe_initial_expiration(block_number: BlockNumber, reveal_timeout: BlockTimeout, lock_timeout: Optional[BlockTimeout] = None) -> BlockExpiration:
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

def is_channel_usable_for_mediation(channel_state: ChannelState, transfer_amount: PaymentAmount, lock_timeout: BlockTimeout) -> bool:
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
    channel_usable = is_channel_usable_for_new_transfer(channel_state, transfer_amount, lock_timeout)
    return channel_usable is ChannelUsability.USABLE

def is_channel_usable_for_new_transfer(channel_state: ChannelState, transfer_amount: PaymentAmount, lock_timeout: Optional[BlockTimeout]) -> ChannelUsability:
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
    lock_timeout_valid = lock_timeout is None or (lock_timeout <= channel_state.settle_timeout and lock_timeout > channel_state.reveal_timeout)
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
    return secrethash in end_state.secrethashes_to_lockedlocks or secrethash in end_state.secrethashes_to_unlockedlocks or secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    """True if the `secrethash` is for a lock with an unknown secret."""
    return secrethash in end_state.secrethashes_to_lockedlocks

def is_lock_expired(end_state: NettingChannelEndState, lock: HashTimeLockState, block_number: BlockNumber, lock_expiration_threshold: BlockExpiration) -> SuccessOrError:
    """Determine whether a lock has expired.

    The lock has expired if both:

        - The secret was not registered on-chain in time.
        - The current block exceeds lock's expiration + confirmation blocks.
    """
    secret_registered_on_chain = lock.secrethash in end_state.secrethashes_to_onchain_unlockedlocks
    if secret_registered_on_chain:
        return SuccessOrError('lock has been unlocked on-chain')
    if block_number < lock_expiration_threshold:
        return SuccessOrError(f'current block number ({block_number}) is not larger than lock.expiration + confirmation blocks ({lock_expiration_threshold})')
    return SuccessOrError()

def is_transfer_expired(transfer: LockedTransferSignedState, affected_channel: ChannelState, block_number: BlockNumber) -> bool:
    lock_expiration_threshold = get_sender_expiration_threshold(transfer.lock.expiration)
    has_lock_expired = is_lock_expired(end_state=affected_channel.our_state, lock=transfer.lock, block_number=block_number, lock_expiration_threshold=lock_expiration_threshold)
    return has_lock_expired.ok

def is_withdraw_expired(block_number: BlockNumber, expiration_threshold: BlockExpiration) -> bool:
    """Determine whether a withdraw has expired.

    The withdraw has expired if the current block exceeds
    the withdraw's expiration + confirmation blocks.
    """
    return block_number >= expiration_threshold

def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    """True if the `secrethash` is for a lock with a known secret."""
    return secrethash in end_state.secrethashes_to_unlockedlocks or secrethash in end_state.secrethashes_to_onchain_unlockedlocks

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

def is_valid_withdraw(withdraw_request: ReceiveWithdrawRequest) -> bool:
    """True if the signature of the message corresponds is valid.

    This predicate is intentionally only checking the signature against the
    message data, and not the expected data. Before this check the fields of
    the message must be validated.
    """
    packed = pack_withdraw(canonical_identifier=withdraw_request.canonical_identifier, participant=withdraw_request.participant, total_withdraw=withdraw_request.total_withdraw, expiration_block=withdraw_request.expiration)
    return is_valid_signature(data=packed, signature=withdraw_request.signature, sender_address=withdraw_request.sender)

def get_secret(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[Secret]:
    """Returns `secret` if the `secrethash` is for a lock with a known secret."""
    partial_unlock_proof = end_state.secrethashes_to_unlockedlocks.get(secrethash)
    if partial_unlock_proof is None:
        partial_unlock_proof = end_state.secrethashes_to_onchain_unlockedlocks.get(secrethash)
    if partial_unlock_proof is not None:
        return partial_unlock_proof.secret
    return None

def is_balance_proof_safe_for_onchain_operations(balance_proof: BalanceProofSignedState) -> bool:
    """Check if the balance proof would overflow onchain."""
    total_amount = balance_proof.transferred_amount + balance_proof.locked_amount
    return total_amount <= UINT256_MAX

def is_valid_amount(end_state: NettingChannelEndState, amount: PaymentAmount) -> bool:
    _, _, current_transferred_amount, current_locked_amount = get_current_balanceproof(end_state)
    transferred_amount_after_unlock = current_transferred_amount + current_locked_amount + amount
    return transferred_amount_after_unlock <= UINT256_MAX

def is_valid_signature(data: bytes, signature: Signature, sender_address: Address) -> SuccessOrError:
    try:
        signer_address = recover(data=data, signature=signature)
    except Exception:
        return SuccessOrError('Signature invalid, could not be recovered.')
    is_correct_sender = sender_address == signer_address
    if is_correct_sender:
        return SuccessOrError()
    return SuccessOrError('Signature was valid but the expected address does not match.')

def is_valid_balanceproof_signature(balance_proof: BalanceProofSignedState, sender_address: Address) -> SuccessOrError:
    balance_hash = hash_balance_data(balance_proof.transferred_amount, balance_proof.locked_amount, balance_proof.locksroot)
    data_that_was_signed = pack_balance_proof(nonce=balance_proof.nonce, balance_hash=balance_hash, additional_hash=balance_proof.message_hash, canonical_identifier=CanonicalIdentifier(chain_identifier=balance_proof.chain_id, token_network_address=balance_proof.token_network_address, channel_identifier=balance_proof.channel_identifier))
    return is_valid_signature(data=data_that_was_signed, signature=balance_proof.signature, sender_address=sender_address)

def is_balance_proof_usable_onchain(received_balance_proof: BalanceProofSignedState, channel_state: ChannelState, sender_state: NettingChannelEndState) -> SuccessOrError:
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
    is_valid_signature = is_valid_balanceproof_signature(received_balance_proof, sender_state.address)
    if get_status(channel_state) != ChannelState.STATE_OPENED:
        return SuccessOrError('The channel is already closed.')
    elif received_balance_proof.channel_identifier != channel_state.identifier:
        return SuccessOrError(f'channel_identifier does not match. expected: {channel_state.identifier} got: {received_balance_proof.channel_identifier}.')
    elif received_balance_proof.token_network_address != channel_state.token_network_address:
        return SuccessOrError(f'token_network_address does not match. expected: