import random
from enum import Enum
from functools import singledispatch
from typing import TYPE_CHECKING
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
from raiden.utils.typing import MYPY_ANNOTATION, Address, AddressMetadata, Any, Balance, BlockExpiration, BlockHash, BlockNumber, BlockTimeout, ChainID, ChannelID, Dict, EncodedData, InitiatorAddress, List, LockedAmount, Locksroot, LockType, MessageID, NamedTuple, Nonce, Optional, PaymentAmount, PaymentID, PaymentWithFeeAmount, Secret, SecretHash, Signature, TargetAddress, TokenAmount, TokenNetworkAddress, Tuple, Union, WithdrawAmount, typecheck
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

def get_safe_initial_expiration(block_number: Union[float, int, raiden.utils.BlockTimeout], reveal_timeout: Union[int, float], lock_timeout: Union[float, int, raiden.utils.BlockTimeout]=None) -> BlockExpiration:
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

def get_sender_expiration_threshold(expiration: Union[int, float, django.utils.timezone.datetime]) -> BlockExpiration:
    """Compute the block at which an expiration message can be sent without
    worrying about blocking the message queue.

    The expiry messages will be rejected if the expiration block has not been
    confirmed. Additionally the sender can account for possible delays in the
    receiver, so a few additional blocks are used to avoid hanging the channel.
    """
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS * 2)

def get_receiver_expiration_threshold(expiration: Union[int, float, None]) -> BlockExpiration:
    """Returns the block number at which the receiver can accept an expiry
    message.

    The receiver must wait for the block at which the expired message to be
    confirmed. This is necessary to handle reorgs, e.g. which could hide a
    secret registration.
    """
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS)

def is_channel_usable_for_mediation(channel_state: Union[raiden.utils.PaymentWithFeeAmount, raiden.utils.BlockTimeout, raiden.utils.Balance], transfer_amount: Union[raiden.utils.PaymentWithFeeAmount, raiden.utils.BlockTimeout, raiden.utils.Balance], lock_timeout: Union[raiden.utils.PaymentWithFeeAmount, raiden.utils.BlockTimeout, raiden.utils.Balance]) -> bool:
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

def is_channel_usable_for_new_transfer(channel_state: Union[str, int, RaidenService], transfer_amount: Union[bool, raiden.utils.PaymentWithFeeAmount, raiden.utils.BlockTimeout], lock_timeout: Union[raiden.utils.BlockTimeout, float, raiden.utils.TokenAmount]):
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

def is_lock_pending(end_state: Union[raiden.transfer.state.NettingChannelEndState, str], secrethash: Union[raiden.transfer.state.NettingChannelEndState, str]) -> bool:
    """True if the `secrethash` corresponds to a lock that is pending to be claimed
    and didn't expire.
    """
    return secrethash in end_state.secrethashes_to_lockedlocks or secrethash in end_state.secrethashes_to_unlockedlocks or secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_lock_locked(end_state: Union[str, bool, NettingChannelEndState], secrethash: Union[str, bool, NettingChannelEndState]) -> bool:
    """True if the `secrethash` is for a lock with an unknown secret."""
    return secrethash in end_state.secrethashes_to_lockedlocks

def is_lock_expired(end_state: Union[raiden.utils.NetworkTimeout, raiden.utils.TokenAmount], lock: Union[raiden.utils.NetworkTimeout, raiden.utils.TokenAmount], block_number: Union[int, float], lock_expiration_threshold: Union[int, float]) -> SuccessOrError:
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

def is_transfer_expired(transfer: Union[raiden.utils.Address, raiden.utils.ChannelMap, raiden.utils.TokenAmount], affected_channel: Union[raiden.utils.Address, int, raiden.utils.BlockNumber, None], block_number: Union[raiden.utils.Address, int, raiden.utils.BlockNumber, None]):
    lock_expiration_threshold = get_sender_expiration_threshold(transfer.lock.expiration)
    has_lock_expired = is_lock_expired(end_state=affected_channel.our_state, lock=transfer.lock, block_number=block_number, lock_expiration_threshold=lock_expiration_threshold)
    return has_lock_expired.ok

def is_withdraw_expired(block_number: Union[int, raiden.utils.TokenAmount, raiden.utils.BlockNumber], expiration_threshold: Union[int, raiden.utils.TokenAmount, raiden.utils.BlockNumber]) -> bool:
    """Determine whether a withdraw has expired.

    The withdraw has expired if the current block exceeds
    the withdraw's expiration + confirmation blocks.
    """
    return block_number >= expiration_threshold

def is_secret_known(end_state: Union[raiden.transfer.state.NettingChannelEndState, str], secrethash: Union[raiden.transfer.state.NettingChannelEndState, str]) -> bool:
    """True if the `secrethash` is for a lock with a known secret."""
    return secrethash in end_state.secrethashes_to_unlockedlocks or secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_secret_known_offchain(end_state: Union[bool, str, NettingChannelEndState], secrethash: Union[bool, str, NettingChannelEndState]) -> bool:
    """True if the `secrethash` is for a lock with a known secret."""
    return secrethash in end_state.secrethashes_to_unlockedlocks

def is_secret_known_onchain(end_state: Union[raiden.transfer.state.NettingChannelEndState, raiden.tests.fuzz.utils.SendSecretRequestInNode], secrethash: Union[raiden.transfer.state.NettingChannelEndState, raiden.tests.fuzz.utils.SendSecretRequestInNode]) -> bool:
    """True if the `secrethash` is for a lock with a known secret."""
    return secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_valid_channel_total_withdraw(channel_total_withdraw: int) -> bool:
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

def is_valid_withdraw(withdraw_request: Union[int, str, raiden.utils.BlockNumber]):
    """True if the signature of the message corresponds is valid.

    This predicate is intentionally only checking the signature against the
    message data, and not the expected data. Before this check the fields of
    the message must be validated.
    """
    packed = pack_withdraw(canonical_identifier=withdraw_request.canonical_identifier, participant=withdraw_request.participant, total_withdraw=withdraw_request.total_withdraw, expiration_block=withdraw_request.expiration)
    return is_valid_signature(data=packed, signature=withdraw_request.signature, sender_address=withdraw_request.sender)

def get_secret(end_state: Union[raiden.transfer.state.NettingChannelEndState, raiden.utils.Any], secrethash: Union[raiden.transfer.state.NettingChannelEndState, raiden.utils.Any]) -> None:
    """Returns `secret` if the `secrethash` is for a lock with a known secret."""
    partial_unlock_proof = end_state.secrethashes_to_unlockedlocks.get(secrethash)
    if partial_unlock_proof is None:
        partial_unlock_proof = end_state.secrethashes_to_onchain_unlockedlocks.get(secrethash)
    if partial_unlock_proof is not None:
        return partial_unlock_proof.secret
    return None

def is_balance_proof_safe_for_onchain_operations(balance_proof: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.utils.TokenAmount]) -> bool:
    """Check if the balance proof would overflow onchain."""
    total_amount = balance_proof.transferred_amount + balance_proof.locked_amount
    return total_amount <= UINT256_MAX

def is_valid_amount(end_state: Union[int, str, NettingChannelEndState], amount: Union[int, float, raiden.utils.BlockTimeout]) -> bool:
    _, _, current_transferred_amount, current_locked_amount = get_current_balanceproof(end_state)
    transferred_amount_after_unlock = current_transferred_amount + current_locked_amount + amount
    return transferred_amount_after_unlock <= UINT256_MAX

def is_valid_signature(data: Union[str, bytes, Address], signature: Union[str, bytes, Address], sender_address: str) -> SuccessOrError:
    try:
        signer_address = recover(data=data, signature=signature)
    except Exception:
        return SuccessOrError('Signature invalid, could not be recovered.')
    is_correct_sender = sender_address == signer_address
    if is_correct_sender:
        return SuccessOrError()
    return SuccessOrError('Signature was valid but the expected address does not match.')

def is_valid_balanceproof_signature(balance_proof: Union[raiden.utils.Address, BalanceProofSignedState, raiden.utils.MessageID], sender_address: Union[raiden.utils.Address, BalanceProofSignedState]):
    balance_hash = hash_balance_data(balance_proof.transferred_amount, balance_proof.locked_amount, balance_proof.locksroot)
    data_that_was_signed = pack_balance_proof(nonce=balance_proof.nonce, balance_hash=balance_hash, additional_hash=balance_proof.message_hash, canonical_identifier=CanonicalIdentifier(chain_identifier=balance_proof.chain_id, token_network_address=balance_proof.token_network_address, channel_identifier=balance_proof.channel_identifier))
    return is_valid_signature(data=data_that_was_signed, signature=balance_proof.signature, sender_address=sender_address)

def is_balance_proof_usable_onchain(received_balance_proof: Union[raiden.utils.MonitoringServiceAddress, raiden.utils.SecretRegistryAddress, raiden.transfer.mediated_transfer.state_change.ActionInitInitiator], channel_state: Union[raiden.transfer.state.ChainState, raiden.utils.SecretHash], sender_state: Union[raiden.transfer.state.ChainState, dict, raiden.transfer.mediated_transfer.state_change.ActionInitInitiator]) -> SuccessOrError:
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
        return SuccessOrError(f'token_network_address does not match. expected: {to_checksum_address(channel_state.token_network_address)} got: {to_checksum_address(received_balance_proof.token_network_address)}.')
    elif received_balance_proof.chain_id != channel_state.chain_id:
        return SuccessOrError(f"chain_id does not match channel's chain_id. expected: {channel_state.chain_id} got: {received_balance_proof.chain_id}.")
    elif not is_balance_proof_safe_for_onchain_operations(received_balance_proof):
        transferred_amount_after_unlock = received_balance_proof.transferred_amount + received_balance_proof.locked_amount
        return SuccessOrError(f'Balance proof total transferred amount would overflow onchain. max: {UINT256_MAX} result would be: {transferred_amount_after_unlock}')
    elif received_balance_proof.nonce != expected_nonce:
        return SuccessOrError(f'Nonce did not change sequentially, expected: {expected_nonce} got: {received_balance_proof.nonce}.')
    else:
        return is_valid_signature

def is_valid_lockedtransfer(transfer_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.state.ChainState, str], channel_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.state.ChainState, str], sender_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.state.ChainState, str], receiver_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.state.ChainState, str]):
    return valid_lockedtransfer_check(channel_state=channel_state, sender_state=sender_state, receiver_state=receiver_state, message_name='LockedTransfer', received_balance_proof=transfer_state.balance_proof, lock=transfer_state.lock)

def is_valid_lock_expired(state_change: Union[raiden.transfer.mediated_transfer.state.LockedTransferUnsignedState, raiden.utils.SecreHash, raiden.transfer.mediated_transfer.state.InitiatorPaymentState], channel_state: Union[raiden.transfer.state.TokenNetworkState, raiden.utils.ChannelMap, raiden.transfer.state_change.ReceiveUnlock], sender_state: Union[raiden.utils.ChannelMap, raiden.transfer.state_change.ReceiveUnlock], receiver_state: Union[raiden.utils.Address, raiden.network.proxies.payment_channel.PaymentChannel, int], block_number: Union[raiden.utils.Address, raiden.network.proxies.payment_channel.PaymentChannel, int]):
    secrethash = state_change.secrethash
    received_balance_proof = state_change.balance_proof
    lock = channel_state.partner_state.secrethashes_to_lockedlocks.get(secrethash)
    if not lock:
        partial_lock = channel_state.partner_state.secrethashes_to_unlockedlocks.get(secrethash)
        if partial_lock:
            lock = partial_lock.lock
    secret_registered_on_chain = secrethash in channel_state.partner_state.secrethashes_to_onchain_unlockedlocks
    current_balance_proof = get_current_balanceproof(sender_state)
    _, _, current_transferred_amount, current_locked_amount = current_balance_proof
    is_valid_balance_proof = is_balance_proof_usable_onchain(received_balance_proof=received_balance_proof, channel_state=channel_state, sender_state=sender_state)
    if lock:
        pending_locks = compute_locks_without(sender_state.pending_locks, EncodedData(bytes(lock.encoded)))
        expected_locked_amount = current_locked_amount - lock.amount
    result = (False, None, None)
    if secret_registered_on_chain:
        msg = 'Invalid LockExpired message. Lock was unlocked on-chain.'
        result = (False, msg, None)
    elif lock is None:
        msg = f'Invalid LockExpired message. Lock with secrethash {to_hex(secrethash)} is not known.'
        result = (False, msg, None)
    elif not is_valid_balance_proof:
        msg = f'Invalid LockExpired message. {is_valid_balance_proof.as_error_message}'
        result = (False, msg, None)
    elif pending_locks is None:
        msg = 'Invalid LockExpired message. Same lock handled twice.'
        result = (False, msg, None)
    else:
        locksroot_without_lock = compute_locksroot(pending_locks)
        check_lock_expired = is_lock_expired(end_state=receiver_state, lock=lock, block_number=block_number, lock_expiration_threshold=get_receiver_expiration_threshold(lock.expiration))
        if not check_lock_expired:
            msg = f'Invalid LockExpired message. {check_lock_expired.as_error_message}'
            result = (False, msg, None)
        elif received_balance_proof.locksroot != locksroot_without_lock:
            msg = "Invalid LockExpired message. Balance proof's locksroot didn't match, expected: {} got: {}.".format(encode_hex(locksroot_without_lock), encode_hex(received_balance_proof.locksroot))
            result = (False, msg, None)
        elif received_balance_proof.transferred_amount != current_transferred_amount:
            msg = "Invalid LockExpired message. Balance proof's transferred_amount changed, expected: {} got: {}.".format(current_transferred_amount, received_balance_proof.transferred_amount)
            result = (False, msg, None)
        elif received_balance_proof.locked_amount != expected_locked_amount:
            msg = "Invalid LockExpired message. Balance proof's locked_amount is invalid, expected: {} got: {}.".format(expected_locked_amount, received_balance_proof.locked_amount)
            result = (False, msg, None)
        else:
            result = (True, None, pending_locks)
    return result

def valid_lockedtransfer_check(channel_state: Union[raiden.transfer.state.NettingChannelState, None, LockedTransferSignedState, raiden.messages.LockedTransfer], sender_state: Union[raiden.transfer.state.NettingChannelState, None, raiden.utils.BlockTimeout, raiden.transfer.state.RouteState], receiver_state: Union[raiden.utils.Address, str, raiden.utils.ChannelID], message_name: Union[str, list], received_balance_proof: Union[raiden.transfer.state.NettingChannelState, None, raiden.transfer.mediated_transfer.state_change.ActionInitInitiator, RequestMonitoring], lock: Union[raiden.transfer.mediated_transfer.state.InitiatorPaymentState, raiden.transfer.architecture.StateChange, random.Random]) -> tuple[typing.Union[bool,None,str]]:
    current_balance_proof = get_current_balanceproof(sender_state)
    pending_locks = compute_locks_with(sender_state.pending_locks, lock)
    _, _, current_transferred_amount, current_locked_amount = current_balance_proof
    distributable = get_distributable(sender_state, receiver_state)
    expected_locked_amount = current_locked_amount + lock.amount
    is_valid_balance_proof = is_balance_proof_usable_onchain(received_balance_proof=received_balance_proof, channel_state=channel_state, sender_state=sender_state)
    result = (False, None, None)
    if not is_valid_balance_proof:
        msg = f'Invalid {message_name} message. {is_valid_balance_proof.as_error_message}'
        result = (False, msg, None)
    elif pending_locks is None:
        msg = f'Invalid {message_name} message. Same lock handled twice.'
        result = (False, msg, None)
    elif len(pending_locks.locks) > MAXIMUM_PENDING_TRANSFERS:
        msg = f'Invalid {message_name} message. Adding the transfer would exceed the allowed limit of {MAXIMUM_PENDING_TRANSFERS} pending transfers per channel.'
        result = (False, msg, None)
    else:
        locksroot_with_lock = compute_locksroot(pending_locks)
        if received_balance_proof.locksroot != locksroot_with_lock:
            msg = "Invalid {} message. Balance proof's locksroot didn't match, expected: {} got: {}.".format(message_name, encode_hex(locksroot_with_lock), encode_hex(received_balance_proof.locksroot))
            result = (False, msg, None)
        elif received_balance_proof.transferred_amount != current_transferred_amount:
            msg = "Invalid {} message. Balance proof's transferred_amount changed, expected: {} got: {}.".format(message_name, current_transferred_amount, received_balance_proof.transferred_amount)
            result = (False, msg, None)
        elif received_balance_proof.locked_amount != expected_locked_amount:
            msg = "Invalid {} message. Balance proof's locked_amount is invalid, expected: {} got: {}.".format(message_name, expected_locked_amount, received_balance_proof.locked_amount)
            result = (False, msg, None)
        elif lock.amount > distributable:
            msg = 'Invalid {} message. Lock amount larger than the available distributable, lock amount: {} maximum distributable: {}'.format(message_name, lock.amount, distributable)
            result = (False, msg, None)
        else:
            result = (True, None, pending_locks)
    return result

def refund_transfer_matches_transfer(refund_transfer: Union[raiden.messages.LockedTransfer, raiden.transfer.state.BalanceProofSignedState, raiden.transfer.mediated_transfer.state_change.TransferDescriptionWithSecretState], transfer: Union[raiden.messages.LockedTransfer, raiden.transfer.mediated_transfer.state_change.TransferDescriptionWithSecretState, raiden.utils.SecretHash]) -> bool:
    refund_transfer_sender = refund_transfer.balance_proof.sender
    if TargetAddress(refund_transfer_sender) == transfer.target:
        return False
    return transfer.payment_identifier == refund_transfer.payment_identifier and transfer.lock.amount == refund_transfer.lock.amount and (transfer.lock.secrethash == refund_transfer.lock.secrethash) and (transfer.target == refund_transfer.target) and (transfer.lock.expiration == refund_transfer.lock.expiration) and (transfer.token == refund_transfer.token)

def is_valid_refund(refund: Union[raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal], channel_state: Union[raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal], sender_state: Union[raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal], receiver_state: Union[raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal], received_transfer: Union[random.Random, raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state_change.ActionInitMediator]) -> Union[tuple[typing.Optional[bool]], tuple[typing.Union[bool,typing.Text,None]], tuple[typing.Union[bool,typing.Text]]]:
    is_valid_locked_transfer, msg, pending_locks = valid_lockedtransfer_check(channel_state, sender_state, receiver_state, 'RefundTransfer', refund.transfer.balance_proof, refund.transfer.lock)
    if not is_valid_locked_transfer:
        return (False, msg, None)
    if not refund_transfer_matches_transfer(refund.transfer, received_transfer):
        return (False, 'Refund transfer did not match the received transfer', None)
    return (True, '', pending_locks)

def is_valid_unlock(unlock: Union[raiden.transfer.state.ChainState, raiden.utils.SecretHash, raiden.transfer.state.NettingChannelState], channel_state: raiden.transfer.state.ChainState, sender_state: raiden.transfer.state.ChainState) -> Union[tuple[typing.Union[bool,str,None]], tuple[typing.Union[bool,typing.Text,None]], tuple[typing.Union[bool,None,raiden.transfer.state.HashTimeLockState]]]:
    received_balance_proof = unlock.balance_proof
    current_balance_proof = get_current_balanceproof(sender_state)
    lock = get_lock(sender_state, unlock.secrethash)
    if lock is None:
        msg = 'Invalid Unlock message. There is no corresponding lock for {}'.format(encode_hex(unlock.secrethash))
        return (False, msg, None)
    pending_locks = compute_locks_without(sender_state.pending_locks, EncodedData(bytes(lock.encoded)))
    if pending_locks is None:
        msg = f'Invalid unlock message. The lock is unknown {encode_hex(lock.encoded)}'
        return (False, msg, None)
    locksroot_without_lock = compute_locksroot(pending_locks)
    _, _, current_transferred_amount, current_locked_amount = current_balance_proof
    expected_transferred_amount = current_transferred_amount + TokenAmount(lock.amount)
    expected_locked_amount = current_locked_amount - lock.amount
    is_valid_balance_proof = is_balance_proof_usable_onchain(received_balance_proof=received_balance_proof, channel_state=channel_state, sender_state=sender_state)
    result = (False, None, None)
    if not is_valid_balance_proof:
        msg = f'Invalid Unlock message. {is_valid_balance_proof.as_error_message}'
        result = (False, msg, None)
    elif received_balance_proof.locksroot != locksroot_without_lock:
        msg = "Invalid Unlock message. Balance proof's locksroot didn't match, expected: {} got: {}.".format(encode_hex(locksroot_without_lock), encode_hex(received_balance_proof.locksroot))
        result = (False, msg, None)
    elif received_balance_proof.transferred_amount != expected_transferred_amount:
        msg = "Invalid Unlock message. Balance proof's wrong transferred_amount, expected: {} got: {}.".format(expected_transferred_amount, received_balance_proof.transferred_amount)
        result = (False, msg, None)
    elif received_balance_proof.locked_amount != expected_locked_amount:
        msg = "Invalid Unlock message. Balance proof's wrong locked_amount, expected: {} got: {}.".format(expected_locked_amount, received_balance_proof.locked_amount)
        result = (False, msg, None)
    else:
        result = (True, None, pending_locks)
    return result

def is_valid_total_withdraw(channel_state: Union[raiden.utils.Address, raiden.transfer.state.NettingChannelState, raiden.utils.TokenAmount], our_total_withdraw: int, allow_zero: bool=False) -> SuccessOrError:
    balance = get_balance(sender=channel_state.our_state, receiver=channel_state.partner_state)
    withdraw_overflow = not is_valid_channel_total_withdraw(TokenAmount(our_total_withdraw + channel_state.partner_total_withdraw))
    withdraw_amount = our_total_withdraw - channel_state.our_total_withdraw
    if get_status(channel_state) != ChannelState.STATE_OPENED:
        return SuccessOrError('Invalid withdraw, the channel is not opened')
    elif withdraw_amount < 0:
        return SuccessOrError(f'Total withdraw {our_total_withdraw} decreased')
    elif not allow_zero and withdraw_amount == 0:
        return SuccessOrError(f'Total withdraw {our_total_withdraw} did not increase')
    elif balance < withdraw_amount:
        return SuccessOrError(f'Insufficient balance: {balance}. Requested {withdraw_amount} for withdraw')
    elif withdraw_overflow:
        return SuccessOrError(f'The new total_withdraw {our_total_withdraw} will cause an overflow')
    else:
        return SuccessOrError()

def is_valid_action_coopsettle(channel_state: Union[raiden.utils.PaymentWithFeeAmount, raiden.utils.BlockTimeout, raiden.utils.TokenAmount], coop_settle: Union[int, typing.Iterable[int]], total_withdraw: Union[raiden.utils.TokenAmount, raiden.utils.PaymentWithFeeAmount, list[int]]) -> Union[int, list, typing.DefaultDict, SuccessOrError]:
    result = is_valid_total_withdraw(channel_state, total_withdraw, allow_zero=True)
    if not result:
        return result
    if get_number_of_pending_transfers(channel_state.our_state) > 0:
        return SuccessOrError('Coop-Settle not allowed: We still have pending locks')
    if get_number_of_pending_transfers(channel_state.partner_state) > 0:
        return SuccessOrError('Coop-Settle not allowed: Partner still has pending locks')
    if channel_state.our_state.offchain_total_withdraw > 0:
        return SuccessOrError('Coop-Settle not allowed: We still have pending withdraws')
    if channel_state.partner_state.offchain_total_withdraw > 0:
        return SuccessOrError('Coop-Settle not allowed: Partner still has pending withdraws')
    return SuccessOrError()

def is_valid_withdraw_request(channel_state: Union[dict[str, typing.Any], raiden.blockchain.events.DecodedEvent, list[raiden.transfer.architecture.Event]], withdraw_request: Union[dict[str, typing.Any], raiden.transfer.mediated_transfer.events.SendSecretRequest, list[raiden.transfer.architecture.Event]]) -> Union[SuccessOrError, bool, str, tuple[str]]:
    expected_nonce = get_next_nonce(channel_state.partner_state)
    balance = get_balance(sender=channel_state.partner_state, receiver=channel_state.our_state)
    is_valid = is_valid_withdraw(withdraw_request)
    withdraw_amount = withdraw_request.total_withdraw - channel_state.partner_total_withdraw
    withdraw_overflow = not is_valid_channel_total_withdraw(TokenAmount(withdraw_request.total_withdraw + channel_state.our_total_withdraw))
    if channel_state.canonical_identifier != withdraw_request.canonical_identifier:
        return SuccessOrError('Invalid canonical identifier provided in withdraw request')
    elif withdraw_request.participant != channel_state.partner_state.address:
        return SuccessOrError('Invalid participant, it must be the partner address')
    elif withdraw_request.sender != channel_state.partner_state.address:
        return SuccessOrError('Invalid sender, withdraw request must be sent by the partner.')
    elif withdraw_amount < 0:
        return SuccessOrError(f'Total withdraw {withdraw_request.total_withdraw} decreased')
    elif balance < withdraw_amount:
        return SuccessOrError(f'Insufficient balance: {balance}. Requested {withdraw_amount} for withdraw')
    elif withdraw_request.nonce != expected_nonce:
        return SuccessOrError(f'Nonce did not change sequentially, expected: {expected_nonce} got: {withdraw_request.nonce}.')
    elif withdraw_overflow:
        return SuccessOrError(f'The new total_withdraw {withdraw_request.total_withdraw} will cause an overflow')
    else:
        return is_valid

def is_valid_withdraw_confirmation(channel_state: Union[raiden.transfer.state.ChainState, raiden.utils.Balance, raiden.transfer.mediated_transfer.events.SendSecretRequest], received_withdraw: Union[raiden.transfer.state.ChainState, raiden.utils.Balance, raiden.transfer.mediated_transfer.events.SendSecretRequest]) -> Union[SuccessOrError, str, bool]:
    withdraw_state = channel_state.our_state.withdraws_pending.get(received_withdraw.total_withdraw)
    if withdraw_state is None:
        try:
            withdraw_state = next((candidate for candidate in channel_state.our_state.withdraws_expired if candidate.total_withdraw == received_withdraw.total_withdraw))
        except StopIteration:
            pass
    expected_nonce = get_next_nonce(channel_state.partner_state)
    if not withdraw_state:
        return SuccessOrError(f'Received withdraw confirmation {received_withdraw.total_withdraw} was not found in withdraw states')
    withdraw_overflow = not is_valid_channel_total_withdraw(TokenAmount(received_withdraw.total_withdraw + channel_state.partner_total_withdraw))
    if channel_state.canonical_identifier != received_withdraw.canonical_identifier:
        return SuccessOrError('Invalid canonical identifier provided in withdraw request')
    elif received_withdraw.total_withdraw != channel_state.our_total_withdraw:
        return SuccessOrError(f'Total withdraw confirmation {received_withdraw.total_withdraw} does not match our total withdraw {channel_state.our_total_withdraw}')
    elif received_withdraw.nonce != expected_nonce:
        return SuccessOrError(f'Nonce did not change sequentially, expected: {expected_nonce} got: {received_withdraw.nonce}.')
    elif received_withdraw.expiration != withdraw_state.expiration:
        return SuccessOrError(f'Invalid expiration {received_withdraw.expiration}, withdraw confirmation must use the same confirmation as the request, otherwise the signatures will not match on-chain.')
    elif received_withdraw.participant != channel_state.our_state.address:
        return SuccessOrError(f'Invalid participant {to_checksum_address(received_withdraw.participant)}, it must be the same as the sender address {to_checksum_address(channel_state.our_state.address)}')
    elif received_withdraw.sender != channel_state.partner_state.address:
        return SuccessOrError('Invalid sender, withdraw confirmation must be sent by the partner.')
    elif withdraw_overflow:
        return SuccessOrError(f'The new total_withdraw {received_withdraw.total_withdraw} will cause an overflow')
    else:
        return is_valid_withdraw(received_withdraw)

def is_valid_withdraw_expired(channel_state: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.utils.ChannelMap], state_change: Union[raiden.utils.Address, raiden.utils.BlockNumber, raiden.utils.TokenAmount], withdraw_state: Union[raiden.utils.TokenAmount, int, raiden.utils.ChannelID], block_number: Union[raiden.utils.TokenAmount, raiden.utils.FeeAmount, int]) -> SuccessOrError:
    expected_nonce = get_next_nonce(channel_state.partner_state)
    withdraw_expired = is_withdraw_expired(block_number=block_number, expiration_threshold=get_receiver_expiration_threshold(expiration=withdraw_state.expiration))
    if not withdraw_expired:
        return SuccessOrError(f'WithdrawExpired for withdraw that has not yet expired {state_change.total_withdraw}.')
    elif channel_state.canonical_identifier != state_change.canonical_identifier:
        return SuccessOrError('Invalid canonical identifier provided in WithdrawExpired')
    elif state_change.sender != channel_state.partner_state.address:
        return SuccessOrError('Expired withdraw not from partner.')
    elif state_change.total_withdraw != withdraw_state.total_withdraw:
        return SuccessOrError(f'WithdrawExpired and local withdraw amounts do not match. Received {state_change.total_withdraw}, local amount {withdraw_state.total_withdraw}')
    elif state_change.nonce != expected_nonce:
        return SuccessOrError(f'Nonce did not change sequentially, expected: {expected_nonce} got: {state_change.nonce}.')
    else:
        return SuccessOrError()

def get_amount_unclaimed_onchain(end_state: Union[raiden.utils.TokenAmount, bool]) -> TokenAmount:
    return TokenAmount(sum((unlock.lock.amount for unlock in end_state.secrethashes_to_onchain_unlockedlocks.values())))

def get_amount_locked(end_state: Union[prefecengine.state.State, bool, raiden.utils.TokenAmount]) -> LockedAmount:
    total_pending = sum((lock.amount for lock in end_state.secrethashes_to_lockedlocks.values()))
    total_unclaimed = sum((unlock.lock.amount for unlock in end_state.secrethashes_to_unlockedlocks.values()))
    total_unclaimed_onchain = get_amount_unclaimed_onchain(end_state)
    result = total_pending + total_unclaimed + total_unclaimed_onchain
    return LockedAmount(result)

def get_capacity(channel_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.state.ChainState]) -> TokenAmount:
    """Calculates the capacity of the given channel

    For the definition of capacity see
    https://raiden-network.readthedocs.io/en/stable/glossary.html#term-Channel-capacity
    """
    return TokenAmount(channel_state.our_total_deposit - channel_state.our_total_withdraw + channel_state.partner_total_deposit - channel_state.partner_total_withdraw)

def get_balance(sender: Union[dict, str, raiden.transfer.state.NettingChannelState], receiver: Union[dict, str, raiden.transfer.state.NettingChannelState]) -> Union[str, int]:
    return _get_balance(sender, receiver, subtract_withdraws=True)

def _get_balance(sender: Union[raiden.utils.Address, raiden.utils.TokenAddress, raiden.utils.PaymentNetworkID], receiver: Union[raiden.utils.PaymentID, raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.transfer.state.BalanceProofSignedState], subtract_withdraws: bool=True) -> Balance:
    """Calculates the balance for a participant in a channel.

    For the definition of balance see
    https://raiden-network.readthedocs.io/en/stable/glossary.html#term-Balance
    """
    sender_transferred_amount = 0
    receiver_transferred_amount = 0
    if sender.balance_proof:
        sender_transferred_amount = sender.balance_proof.transferred_amount
    if receiver.balance_proof:
        receiver_transferred_amount = receiver.balance_proof.transferred_amount
    max_withdraw = max(sender.offchain_total_withdraw, sender.onchain_total_withdraw)
    withdraw = max_withdraw if subtract_withdraws else 0
    return Balance(sender.contract_balance - withdraw - sender_transferred_amount + receiver_transferred_amount)

def get_max_withdraw_amount(sender: Union[raiden.utils.Address, bytes, raiden.utils.TokenAmount], receiver: Union[raiden.utils.Address, bytes, raiden.utils.TokenAmount]) -> WithdrawAmount:
    """Calculates the maximum "total_withdraw_amount" for a channel.
    This will leave the channel without funds, when this is withdrawn from the contract,
    or pending as offchain-withdraw.

    """
    return WithdrawAmount(_get_balance(sender, receiver, subtract_withdraws=False))

def get_current_balanceproof(end_state: Union[NettingChannelEndState, str, bool]) -> tuple[typing.Union[Locksroot,Nonce,TokenAmount,LockedAmount,int,raiden.utils.BlockExpiration]]:
    balance_proof = end_state.balance_proof
    if balance_proof:
        locksroot = balance_proof.locksroot
        nonce = end_state.nonce
        transferred_amount = balance_proof.transferred_amount
        locked_amount = get_amount_locked(end_state)
    else:
        locksroot = Locksroot(LOCKSROOT_OF_NO_LOCKS)
        nonce = Nonce(0)
        transferred_amount = TokenAmount(0)
        locked_amount = LockedAmount(0)
    return (locksroot, nonce, transferred_amount, locked_amount)

def get_current_nonce(end_state: Union[int, NettingChannelEndState]):
    return end_state.nonce

def get_distributable(sender: Union[str, raiden.utils.Address, raiden.utils.TokenNetworkAddress], receiver: Union[str, int]) -> TokenAmount:
    """Return the amount of tokens that can be used by the `sender`.

    The returned value is limited to a UINT256, since that is the representation
    used in the smart contracts and we cannot use a larger value. The limit is
    enforced on transferred_amount + locked_amount to avoid overflows. This is
    an additional security check.
    """
    _, _, transferred_amount, locked_amount = get_current_balanceproof(sender)
    distributable = get_balance(sender, receiver) - get_amount_locked(sender)
    overflow_limit = max(UINT256_MAX - transferred_amount - locked_amount, 0)
    return TokenAmount(min(overflow_limit, distributable))

def get_lock(end_state: Union[raiden.utils.PaymentAmount, raiden.transfer.state.NettingChannelEndState, raiden.utils.Any], secrethash: Union[raiden.utils.PaymentAmount, raiden.transfer.state.NettingChannelEndState, raiden.utils.Any]) -> Union[None, raiden.transfer.state.HashTimeLockState, raiden.messages.Lock]:
    """Return the lock corresponding to `secrethash` or None if the lock is
    unknown.
    """
    lock = end_state.secrethashes_to_lockedlocks.get(secrethash)
    if not lock:
        partial_unlock = end_state.secrethashes_to_unlockedlocks.get(secrethash)
        if not partial_unlock:
            partial_unlock = end_state.secrethashes_to_onchain_unlockedlocks.get(secrethash)
        if partial_unlock:
            lock = partial_unlock.lock
    if lock is not None:
        typecheck(lock, HashTimeLockState)
    return lock

def lock_exists_in_either_channel_side(channel_state: Union[raiden.utils.Address, bytes, raiden.transfer.state.NettingChannelState], secrethash: Union[raiden.utils.Address, bytes, raiden.transfer.state.NettingChannelState]) -> bool:
    """Check if the lock with `secrethash` exists in either our state or the partner's state"""
    lock = get_lock(channel_state.our_state, secrethash)
    if not lock:
        lock = get_lock(channel_state.partner_state, secrethash)
    return lock is not None

def get_next_nonce(end_state: Union[int, bytes]) -> Nonce:
    return Nonce(end_state.nonce + 1)

def get_number_of_pending_transfers(channel_end_state: Union[str, list[int], int, None]) -> int:
    return len(channel_end_state.pending_locks.locks)

def get_status(channel_state: Union[raiden.transfer.state.NettingChannelState, int, raiden.transfer.state.RouteState]):
    if channel_state.settle_transaction:
        finished_successfully = channel_state.settle_transaction.result == TransactionExecutionStatus.SUCCESS
        running = channel_state.settle_transaction.finished_block_number is None
        if finished_successfully:
            result = ChannelState.STATE_SETTLED
        elif running:
            result = ChannelState.STATE_SETTLING
        else:
            result = ChannelState.STATE_UNUSABLE
    elif channel_state.close_transaction:
        finished_successfully = channel_state.close_transaction.result == TransactionExecutionStatus.SUCCESS
        running = channel_state.close_transaction.finished_block_number is None
        if finished_successfully:
            result = ChannelState.STATE_CLOSED
        elif running:
            result = ChannelState.STATE_CLOSING
        else:
            result = ChannelState.STATE_UNUSABLE
    else:
        result = ChannelState.STATE_OPENED
    return result

def _del_unclaimed_lock(end_state: Union[str, NettingChannelEndState, raiden.transfer.architecture.TransitionResult], secrethash: Union[str, NettingChannelEndState, raiden.transfer.architecture.TransitionResult]) -> None:
    if secrethash in end_state.secrethashes_to_lockedlocks:
        del end_state.secrethashes_to_lockedlocks[secrethash]
    if secrethash in end_state.secrethashes_to_unlockedlocks:
        del end_state.secrethashes_to_unlockedlocks[secrethash]

def _del_lock(end_state: Union[raiden.transfer.state.NettingChannelEndState, raiden.utils.TokenAmount], secrethash: Union[raiden.transfer.state.NettingChannelEndState, raiden.utils.TokenAmount]) -> None:
    """Removes the lock from the indexing structures.

    Note:
        This won't change the pending locks!
    """
    assert is_lock_pending(end_state, secrethash), 'Lock must be pending'
    _del_unclaimed_lock(end_state, secrethash)
    if secrethash in end_state.secrethashes_to_onchain_unlockedlocks:
        del end_state.secrethashes_to_onchain_unlockedlocks[secrethash]

def set_closed(channel_state: Union[raiden.utils.Address, raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification], block_number: Union[raiden.utils.BlockSpecification, raiden.utils.Address, raiden.utils.BlockIdentifier]) -> None:
    if not channel_state.close_transaction:
        channel_state.close_transaction = TransactionExecutionStatus(None, block_number, TransactionExecutionStatus.SUCCESS)
    elif not channel_state.close_transaction.finished_block_number:
        channel_state.close_transaction.finished_block_number = block_number
        channel_state.close_transaction.result = TransactionExecutionStatus.SUCCESS

def set_settled(channel_state: Union[raiden.utils.BlockSpecification, raiden.transfer.identifiers.CanonicalIdentifier, raiden.utils.ChannelMap], block_number: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification, raiden.utils.ChannelMap]) -> None:
    if not channel_state.settle_transaction:
        channel_state.settle_transaction = TransactionExecutionStatus(None, block_number, TransactionExecutionStatus.SUCCESS)
    elif not channel_state.settle_transaction.finished_block_number:
        channel_state.settle_transaction.finished_block_number = block_number
        channel_state.settle_transaction.result = TransactionExecutionStatus.SUCCESS

def set_coop_settled(channel_end_state: Union[raiden.utils.Address, raiden.utils.FeeAmount], block_number: Union[raiden.utils.TokenAmount, metrics_backend.utils.Address, raiden.utils.BlockSpecification]) -> None:
    msg = 'This should only be called on a state where a CoopSettle is initiated'
    assert channel_end_state.initiated_coop_settle is not None, msg
    if not channel_end_state.initiated_coop_settle.transaction:
        channel_end_state.initiated_coop_settle.transaction = TransactionExecutionStatus(None, block_number, TransactionExecutionStatus.SUCCESS)
    elif not channel_end_state.initiated_coop_settle.transaction.finished_block_number:
        channel_end_state.initiated_coop_settle.transaction.finished_block_number = block_number
        channel_end_state.initiated_coop_settle.transaction.result = TransactionExecutionStatus.SUCCESS

def update_contract_balance(end_state: Union[bool, raiden.utils.Any, RaidenService], contract_balance: Union[bool, raiden.utils.Any, RaidenService]) -> None:
    if contract_balance > end_state.contract_balance:
        end_state.contract_balance = contract_balance

def compute_locks_with(locks: bool, lock: bool) -> Union[PendingLocksState, None]:
    """Register the given lock with as a pending locks."""
    if bytes(lock.encoded) not in locks.locks:
        locks = PendingLocksState(list(locks.locks))
        locks.locks.append(EncodedData(bytes(lock.encoded)))
        return locks
    else:
        return None

def compute_locks_without(locks: Union[str, bool, list[str]], lock_encoded: Union[bool, str]) -> Union[PendingLocksState, None]:
    if lock_encoded in locks.locks:
        locks = PendingLocksState(list(locks.locks))
        locks.locks.remove(lock_encoded)
        return locks
    else:
        return None

def compute_locksroot(locks: Union[bool, raiden.utils.Address, tuple[typing.Union[bool,str,None]]]) -> Locksroot:
    """Compute the hash representing all pending locks
    The hash is submitted in TokenNetwork.settleChannel() call.
    """
    return Locksroot(keccak(b''.join(locks.locks)))

def create_sendlockedtransfer(channel_state: Union[raiden.utils.TokenAmount, raiden.utils.PaymentAmount, raiden.utils.TokenNetworkID], initiator: Union[raiden.utils.Address, raiden.utils.TokenAddress, raiden.utils.PaymentNetworkID], target: Union[raiden.utils.Address, raiden.utils.TokenAddress, raiden.utils.PaymentNetworkID], amount: Union[raiden.utils.TokenAmount, raiden.utils.PaymentID, float], message_identifier: Union[raiden.utils.Address, raiden.utils.TokenAmount, raiden.utils.SecretHash], payment_identifier: Union[raiden.utils.Address, raiden.utils.TokenAddress, raiden.utils.PaymentNetworkID], expiration: Union[float, raiden.utils.TokenAmount, raiden.utils.PaymentAmount], secret: Union[raiden.utils.Address, raiden.utils.TokenAddress, raiden.utils.PaymentNetworkID], secrethash: Union[float, raiden.utils.TokenAmount, raiden.utils.PaymentAmount], route_states: Union[raiden.utils.Address, raiden.utils.TokenAddress, raiden.utils.PaymentNetworkID], recipient_metadata: Union[None, int, TransactionExecutionStatus, raiden.utils.PaymentID]=None, previous_metadata: Union[None, raiden.utils.Address, raiden.utils.TokenAddress, raiden.utils.PaymentNetworkID]=None) -> tuple[SendLockedTransfer]:
    our_state = channel_state.our_state
    partner_state = channel_state.partner_state
    our_balance_proof = our_state.balance_proof
    msg = 'caller must make sure there is enough balance'
    assert amount <= get_distributable(our_state, partner_state), msg
    msg = 'caller must make sure the channel is open'
    assert get_status(channel_state) == ChannelState.STATE_OPENED, msg
    lock = HashTimeLockState(amount=amount, expiration=expiration, secrethash=secrethash)
    pending_locks = compute_locks_with(channel_state.our_state.pending_locks, lock)
    assert pending_locks, 'lock is already registered'
    locksroot = compute_locksroot(pending_locks)
    if our_balance_proof:
        transferred_amount = our_balance_proof.transferred_amount
    else:
        transferred_amount = TokenAmount(0)
    msg = 'caller must make sure the result wont overflow'
    assert transferred_amount + amount <= UINT256_MAX, msg
    token = channel_state.token_address
    locked_amount = LockedAmount(get_amount_locked(our_state) + amount)
    nonce = get_next_nonce(channel_state.our_state)
    balance_proof = BalanceProofUnsignedState(nonce=nonce, transferred_amount=transferred_amount, locked_amount=locked_amount, locksroot=locksroot, canonical_identifier=channel_state.canonical_identifier)
    locked_transfer = LockedTransferUnsignedState(payment_identifier=payment_identifier, token=token, balance_proof=balance_proof, secret=secret, lock=lock, initiator=initiator, target=target, route_states=route_states, metadata=previous_metadata)
    recipient = channel_state.partner_state.address
    if recipient_metadata is None:
        recipient_metadata = get_address_metadata(recipient, route_states)
    lockedtransfer = SendLockedTransfer(recipient=recipient, recipient_metadata=recipient_metadata, message_identifier=message_identifier, transfer=locked_transfer, canonical_identifier=channel_state.canonical_identifier)
    return (lockedtransfer, pending_locks)

def create_unlock(channel_state: Union[int, raiden.utils.Signature, raiden.messages.LockedTransfer], message_identifier: Union[raiden.utils.TokenNetworkID, raiden.utils.Address, raiden.messages.LockedTransfer], payment_identifier: Union[raiden.utils.TokenNetworkID, raiden.utils.Address, raiden.messages.LockedTransfer], secret: Union[raiden.utils.TokenNetworkID, raiden.utils.Address, raiden.messages.LockedTransfer], lock: Union[raiden.utils.Address, raiden.messages.LockedTransfer, raiden.network.proxies.payment_channel.PaymentChannel], block_number: Union[int, float], recipient_metadata: Union[None, raiden.utils.TokenNetworkID, raiden.utils.Address, raiden.messages.LockedTransfer]=None) -> tuple[SendUnlock]:
    our_state = channel_state.our_state
    msg = 'caller must make sure the lock is known'
    assert is_lock_pending(our_state, lock.secrethash), msg
    msg = 'caller must make sure the channel is open'
    assert get_status(channel_state) == ChannelState.STATE_OPENED, msg
    expired = is_lock_expired(end_state=channel_state.our_state, lock=lock, block_number=block_number, lock_expiration_threshold=lock.expiration)
    msg = 'caller must make sure the lock is not expired'
    assert not expired, msg
    our_balance_proof = our_state.balance_proof
    msg = 'the lock is pending, it must be in the pending locks'
    assert our_balance_proof is not None, msg
    transferred_amount = TokenAmount(lock.amount + our_balance_proof.transferred_amount)
    pending_locks = compute_locks_without(our_state.pending_locks, EncodedData(bytes(lock.encoded)))
    msg = 'the lock is pending, it must be in the pending locks'
    assert pending_locks is not None, msg
    locksroot = compute_locksroot(pending_locks)
    token_address = channel_state.token_address
    recipient = channel_state.partner_state.address
    locked_amount = LockedAmount(get_amount_locked(our_state) - lock.amount)
    nonce = get_next_nonce(our_state)
    channel_state.our_state.nonce = nonce
    balance_proof = BalanceProofUnsignedState(nonce=nonce, transferred_amount=transferred_amount, locked_amount=locked_amount, locksroot=locksroot, canonical_identifier=channel_state.canonical_identifier)
    unlock_lock = SendUnlock(recipient=recipient, recipient_metadata=recipient_metadata, message_identifier=message_identifier, payment_identifier=payment_identifier, token_address=token_address, secret=secret, balance_proof=balance_proof, canonical_identifier=channel_state.canonical_identifier)
    return (unlock_lock, pending_locks)

def send_lockedtransfer(channel_state: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], initiator: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], target: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], amount: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], message_identifier: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], payment_identifier: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], expiration: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], secret: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], secrethash: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], route_states: Union[raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress], recipient_metadata: Union[None, raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress]=None, previous_metadata: Union[None, raiden.utils.PaymentAmount, raiden.utils.PaymentID, raiden.utils.TokenAddress]=None) -> Union[ReceiveLockExpired, raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal]:
    send_locked_transfer_event, pending_locks = create_sendlockedtransfer(channel_state=channel_state, initiator=initiator, target=target, amount=amount, message_identifier=message_identifier, payment_identifier=payment_identifier, expiration=expiration, secret=secret, secrethash=secrethash, route_states=route_states, recipient_metadata=recipient_metadata, previous_metadata=previous_metadata)
    transfer = send_locked_transfer_event.transfer
    lock = transfer.lock
    channel_state.our_state.balance_proof = transfer.balance_proof
    channel_state.our_state.nonce = transfer.balance_proof.nonce
    channel_state.our_state.pending_locks = pending_locks
    channel_state.our_state.secrethashes_to_lockedlocks[lock.secrethash] = lock
    return send_locked_transfer_event

def send_unlock(channel_state: Union[raiden.messages.LockedTransfer, raiden.utils.Address, raiden.utils.InitiatorAddress], message_identifier: Union[raiden.messages.LockedTransfer, raiden.utils.Address, raiden.utils.BlockNumber], payment_identifier: Union[raiden.messages.LockedTransfer, raiden.utils.Address, raiden.utils.BlockNumber], secret: Union[raiden.messages.LockedTransfer, raiden.utils.Address, raiden.utils.BlockNumber], secrethash: Union[raiden.utils.Address, raiden.utils.BlockNumber, raiden.messages.LockedTransfer], block_number: Union[raiden.messages.LockedTransfer, raiden.utils.Address, raiden.utils.BlockNumber], recipient_metadata: Union[None, raiden.messages.LockedTransfer, raiden.utils.Address, raiden.utils.BlockNumber]=None):
    lock = get_lock(channel_state.our_state, secrethash)
    assert lock, 'caller must ensure the lock exists'
    unlock, pending_locks = create_unlock(channel_state, message_identifier, payment_identifier, secret, lock, block_number, recipient_metadata)
    channel_state.our_state.balance_proof = unlock.balance_proof
    channel_state.our_state.pending_locks = pending_locks
    _del_lock(channel_state.our_state, lock.secrethash)
    return unlock

def events_for_close(channel_state: Union[int, raiden.utils.BlockNumber, raiden.utils.BlockIdentifier], block_number: Union[raiden.utils.Address, raiden.transfer.state.TokenNetworkState, raiden.utils.ChannelMap], block_hash: Union[raiden.utils.BlockSpecification, raiden.utils.Address, raiden.transfer.state.NettingChannelState]) -> list[ContractSendChannelClose]:
    events = []
    if get_status(channel_state) in CHANNEL_STATES_PRIOR_TO_CLOSED:
        channel_state.close_transaction = TransactionExecutionStatus(block_number, None, None)
        balance_proof = channel_state.partner_state.balance_proof
        assert balance_proof is None or isinstance(balance_proof, BalanceProofSignedState), 'BP is not signed'
        close_event = ContractSendChannelClose(canonical_identifier=channel_state.canonical_identifier, balance_proof=balance_proof, triggered_by_block_hash=block_hash)
        events.append(close_event)
    return events

def send_withdraw_request(channel_state: Union[int, raiden.transfer.state.NettingChannelState, raiden.utils.BlockNumber], total_withdraw: Union[raiden.utils.ChannelMap, raiden.utils.BlockNumber, int], expiration: Union[raiden.utils.TokenAmount, raiden.utils.Address, int], pseudo_random_generator: Union[raiden.transfer.mediated_transfer.state.MediatorTransferState, raiden.transfer.mediated_transfer.state_change.ReceiveTransferRefund, raiden.transfer.state.TokenNetworkState], recipient_metadata: Union[raiden.utils.TokenAmount, raiden.utils.Address, int], coop_settle: bool=False) -> Union[list, list[SendWithdrawRequest]]:
    events = []
    if get_status(channel_state) not in CHANNEL_STATES_PRIOR_TO_CLOSED:
        return events
    nonce = get_next_nonce(channel_state.our_state)
    withdraw_state = PendingWithdrawState(total_withdraw=total_withdraw, nonce=nonce, expiration=expiration, recipient_metadata=recipient_metadata)
    channel_state.our_state.nonce = nonce
    channel_state.our_state.withdraws_pending[withdraw_state.total_withdraw] = withdraw_state
    withdraw_event = SendWithdrawRequest(canonical_identifier=CanonicalIdentifier(chain_identifier=channel_state.chain_id, token_network_address=channel_state.token_network_address, channel_identifier=channel_state.identifier), recipient=channel_state.partner_state.address, message_identifier=message_identifier_from_prng(pseudo_random_generator), total_withdraw=withdraw_state.total_withdraw, participant=channel_state.our_state.address, nonce=channel_state.our_state.nonce, expiration=withdraw_state.expiration, recipient_metadata=recipient_metadata, coop_settle=coop_settle)
    events.append(withdraw_event)
    return events

def create_sendexpiredlock(sender_end_state: Union[random.Random, int, raiden.utils.BlockNumber], locked_lock: Union[int, random.Random, raiden.utils.Address], pseudo_random_generator: Union[raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state.InitiatorTransferState, bool], chain_id: Union[random.Random, int, raiden.utils.BlockNumber], token_network_address: Union[random.Random, int, raiden.utils.BlockNumber], channel_identifier: Union[random.Random, int, raiden.utils.BlockNumber], recipient: Union[raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state.InitiatorTransferState, bool], recipient_metadata: Union[None, raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state.InitiatorTransferState, bool]=None) -> Union[tuple[None], tuple[SendLockExpired]]:
    locked_amount = get_amount_locked(sender_end_state)
    balance_proof = sender_end_state.balance_proof
    updated_locked_amount = LockedAmount(locked_amount - locked_lock.amount)
    assert balance_proof is not None, 'there should be a balance proof because a lock is expiring'
    transferred_amount = balance_proof.transferred_amount
    pending_locks = compute_locks_without(sender_end_state.pending_locks, EncodedData(bytes(locked_lock.encoded)))
    if not pending_locks:
        return (None, None)
    nonce = get_next_nonce(sender_end_state)
    locksroot = compute_locksroot(pending_locks)
    balance_proof = BalanceProofUnsignedState(nonce=nonce, transferred_amount=transferred_amount, locked_amount=updated_locked_amount, locksroot=locksroot, canonical_identifier=CanonicalIdentifier(chain_identifier=chain_id, token_network_address=token_network_address, channel_identifier=channel_identifier))
    send_lock_expired = SendLockExpired(recipient=recipient, recipient_metadata=recipient_metadata, canonical_identifier=balance_proof.canonical_identifier, message_identifier=message_identifier_from_prng(pseudo_random_generator), balance_proof=balance_proof, secrethash=locked_lock.secrethash)
    return (send_lock_expired, pending_locks)

def send_lock_expired(channel_state: Union[raiden.utils.BlockNumber, raiden.transfer.state.NettingChannelState], locked_lock: raiden.utils.BlockNumber, pseudo_random_generator: raiden.utils.BlockNumber, recipient_metadata: Union[None, raiden.utils.BlockNumber]=None) -> list:
    msg = 'caller must make sure the channel is open'
    assert get_status(channel_state) == ChannelState.STATE_OPENED, msg
    send_lock_expired, pending_locks = create_sendexpiredlock(sender_end_state=channel_state.our_state, locked_lock=locked_lock, pseudo_random_generator=pseudo_random_generator, chain_id=channel_state.chain_id, token_network_address=channel_state.token_network_address, channel_identifier=channel_state.identifier, recipient=channel_state.partner_state.address, recipient_metadata=recipient_metadata)
    if send_lock_expired:
        assert pending_locks, 'create_sendexpiredlock should return both message and pending locks'
        channel_state.our_state.pending_locks = pending_locks
        channel_state.our_state.balance_proof = send_lock_expired.balance_proof
        channel_state.our_state.nonce = send_lock_expired.balance_proof.nonce
        _del_unclaimed_lock(channel_state.our_state, locked_lock.secrethash)
        return [send_lock_expired]
    return []

def events_for_expired_withdraws(channel_state: Union[list[raiden.utils.Address], utils.SimpleReachabilityContainer, raiden.transfer.mediated_transfer.state.MediatorTransferState], block_number: Union[int, raiden.utils.TokenAmount, raiden.utils.Signature], pseudo_random_generator: Union[raiden.transfer.mediated_transfer.state.MediatorTransferState, list[raiden.utils.Address], raiden.utils.BlockNumber]) -> list[SendWithdrawExpired]:
    events = []
    for withdraw_state in list(channel_state.our_state.withdraws_pending.values()):
        withdraw_expired = is_withdraw_expired(block_number=block_number, expiration_threshold=get_sender_expiration_threshold(withdraw_state.expiration))
        if not withdraw_expired:
            break
        nonce = get_next_nonce(channel_state.our_state)
        channel_state.our_state.nonce = nonce
        coop_settle = channel_state.our_state.initiated_coop_settle
        if coop_settle is not None:
            if coop_settle.total_withdraw_initiator == withdraw_state.total_withdraw and coop_settle.expiration == withdraw_state.expiration:
                channel_state.our_state.initiated_coop_settle = None
        channel_state.our_state.withdraws_expired.append(ExpiredWithdrawState(withdraw_state.total_withdraw, withdraw_state.expiration, withdraw_state.nonce, withdraw_state.recipient_metadata))
        del channel_state.our_state.withdraws_pending[withdraw_state.total_withdraw]
        events.append(SendWithdrawExpired(recipient=channel_state.partner_state.address, recipient_metadata=withdraw_state.recipient_metadata, canonical_identifier=channel_state.canonical_identifier, message_identifier=message_identifier_from_prng(pseudo_random_generator), total_withdraw=withdraw_state.total_withdraw, participant=channel_state.our_state.address, expiration=withdraw_state.expiration, nonce=nonce))
    return events

def register_secret_endstate(end_state: Union[raiden.storage.sqlite.SerializedSQLiteStorage, bool], secret: Union[raiden.messages.LockedTransfer, bool, raiden.utils.Secret], secrethash: Union[raiden.storage.sqlite.SerializedSQLiteStorage, bool]) -> None:
    if is_lock_locked(end_state, secrethash):
        pending_lock = end_state.secrethashes_to_lockedlocks[secrethash]
        del end_state.secrethashes_to_lockedlocks[secrethash]
        end_state.secrethashes_to_unlockedlocks[secrethash] = UnlockPartialProofState(pending_lock, secret)

def register_onchain_secret_endstate(end_state: Union[raiden.utils.Address, raiden.utils.NetworkTimeout, raiden.network.proxies.payment_channel.PaymentChannel], secret: Union[raiden.utils.NetworkTimeout, raiden.transfer.state.HashTimeLockState, raiden.utils.TokenNetworkID], secrethash: Union[raiden.utils.Address, raiden.utils.NetworkTimeout, raiden.network.proxies.payment_channel.PaymentChannel], secret_reveal_block_number: Union[int, raiden.utils.TokenAmount, raiden.utils.Locksroot], delete_lock: bool=True) -> None:
    pending_lock = None
    if is_lock_locked(end_state, secrethash):
        pending_lock = end_state.secrethashes_to_lockedlocks[secrethash]
    if secrethash in end_state.secrethashes_to_unlockedlocks:
        pending_lock = end_state.secrethashes_to_unlockedlocks[secrethash].lock
    if pending_lock:
        if pending_lock.expiration < secret_reveal_block_number:
            return
        if delete_lock:
            _del_lock(end_state, secrethash)
        end_state.secrethashes_to_onchain_unlockedlocks[secrethash] = UnlockPartialProofState(pending_lock, secret)

def register_offchain_secret(channel_state: Union[raiden.transfer.state.ChainState, raiden.utils.ChannelMap, raiden.utils.BlockNumber], secret: Union[raiden.utils.Secret, raiden.tests.fuzz.utils.SendSecretRevealInNode, raiden.utils.SecreHash], secrethash: Union[raiden.utils.Secret, raiden.tests.fuzz.utils.SendSecretRevealInNode, raiden.utils.SecreHash]) -> None:
    """This will register the secret and set the lock to the unlocked stated.

    Even though the lock is unlock it is *not* claimed. The capacity will
    increase once the next balance proof is received.
    """
    our_state = channel_state.our_state
    partner_state = channel_state.partner_state
    register_secret_endstate(our_state, secret, secrethash)
    register_secret_endstate(partner_state, secret, secrethash)

def register_onchain_secret(channel_state: Union[raiden.utils.TokenAmount, raiden.utils.BlockIdentifier, raiden.utils.Address], secret: Union[raiden.utils.Address, list[raiden.transfer.state.RouteState], raiden.messages.SecretRequest], secrethash: Union[raiden.utils.Address, list[raiden.transfer.state.RouteState], raiden.messages.SecretRequest], secret_reveal_block_number: Union[raiden.utils.Address, list[raiden.transfer.state.RouteState], raiden.messages.SecretRequest], delete_lock: bool=True) -> None:
    """This will register the onchain secret and set the lock to the unlocked stated.

    Even though the lock is unlocked it is *not* claimed. The capacity will
    increase once the next balance proof is received.
    """
    our_state = channel_state.our_state
    partner_state = channel_state.partner_state
    register_onchain_secret_endstate(our_state, secret, secrethash, secret_reveal_block_number, delete_lock)
    register_onchain_secret_endstate(partner_state, secret, secrethash, secret_reveal_block_number, delete_lock)

@singledispatch
def handle_state_transitions(action: int, channel_state: Union[raiden.transfer.state.TokenNetworkState, raiden.utils.BlockNumber, raiden.utils.BlockHash], block_number: int, block_hash: int, pseudo_random_generator: int) -> TransitionResult:
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_action_close(action: Union[int, raiden.transfer.state.NettingChannelState, raiden.utils.Locksroot], channel_state: Union[int, metrics_backend.utils.Address, raiden.utils.BlockIdentifier], block_number: Union[raiden.transfer.state.NettingChannelState, raiden.utils.BlockIdentifier, raiden.utils.TokenAmount], block_hash: Union[raiden.transfer.state.NettingChannelState, raiden.utils.BlockIdentifier, raiden.utils.TokenAmount], **kwargs) -> TransitionResult:
    msg = 'caller must make sure the ids match'
    assert channel_state.identifier == action.channel_identifier, msg
    events = events_for_close(channel_state=channel_state, block_number=block_number, block_hash=block_hash)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_action_coop_settle(action: Union[raiden.utils.Address, raiden.utils.BlockNumber, raiden.transfer.state.TokenNetworkState], channel_state: Union[int, raiden.utils.BlockNumber, raiden.utils.Address], pseudo_random_generator: Union[int, RaidenService, raiden.utils.BlockIdentifier], block_number: Union[raiden.utils.Address, raiden.utils.BlockNumber, RaidenService], **kwargs) -> TransitionResult:
    events = []
    our_max_total_withdraw = get_max_withdraw_amount(channel_state.our_state, channel_state.partner_state)
    partner_max_total_withdraw = get_max_withdraw_amount(channel_state.partner_state, channel_state.our_state)
    valid_coop_settle = is_valid_action_coopsettle(channel_state, action, our_max_total_withdraw)
    if valid_coop_settle:
        expiration = get_safe_initial_expiration(block_number=block_number, reveal_timeout=channel_state.reveal_timeout)
        coop_settle = CoopSettleState(total_withdraw_initiator=our_max_total_withdraw, total_withdraw_partner=partner_max_total_withdraw, expiration=expiration)
        channel_state.our_state.initiated_coop_settle = coop_settle
        events = send_withdraw_request(channel_state=channel_state, total_withdraw=coop_settle.total_withdraw_initiator, expiration=coop_settle.expiration, pseudo_random_generator=pseudo_random_generator, recipient_metadata=action.recipient_metadata, coop_settle=True)
    else:
        error_msg = valid_coop_settle.as_error_message
        assert error_msg, 'is_valid_action_coopsettle should return error msg if not valid'
        events = [EventInvalidActionCoopSettle(attempted_withdraw=our_max_total_withdraw, reason=error_msg)]
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_action_withdraw(action: Union[raiden.utils.BlockNumber, raiden.utils.Address, raiden.utils.BlockSpecification], channel_state: Union[raiden.utils.Address, int, raiden.utils.BlockNumber], pseudo_random_generator: Union[int, RaidenService, raiden.transfer.state.NettingChannelState], block_number: Union[raiden.utils.Address, raiden.utils.BlockNumber, int], **kwargs) -> TransitionResult:
    events = []
    is_valid_withdraw = is_valid_total_withdraw(channel_state, action.total_withdraw)
    if is_valid_withdraw:
        expiration = get_safe_initial_expiration(block_number=block_number, reveal_timeout=channel_state.reveal_timeout)
        events = send_withdraw_request(channel_state=channel_state, total_withdraw=action.total_withdraw, expiration=expiration, pseudo_random_generator=pseudo_random_generator, recipient_metadata=action.recipient_metadata)
    else:
        error_msg = is_valid_withdraw.as_error_message
        assert error_msg, 'is_valid_total_withdraw should return error msg if not valid'
        events = [EventInvalidActionWithdraw(attempted_withdraw=action.total_withdraw, reason=error_msg)]
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_action_set_reveal_timeout(action: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.mediated_transfer.state.InitiatorPaymentState, int], channel_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.mediated_transfer.state.InitiatorPaymentState, int], **kwargs) -> TransitionResult:
    events = []
    is_valid_reveal_timeout = action.reveal_timeout >= 7 and channel_state.settle_timeout >= action.reveal_timeout * 2
    if is_valid_reveal_timeout:
        channel_state.reveal_timeout = action.reveal_timeout
    else:
        error_msg = 'Settle timeout should be at least twice as large as reveal timeout'
        events = [EventInvalidActionSetRevealTimeout(reveal_timeout=action.reveal_timeout, reason=error_msg)]
    return TransitionResult(channel_state, events)

def events_for_coop_settle(channel_state: Union[raiden.utils.Address, raiden.transfer.state.TokenNetworkState, raiden.transfer.state_change.ReceiveUnlock], coop_settle_state: Union[raiden.utils.Address, raiden.transfer.state.TokenNetworkState, raiden.transfer.state_change.ReceiveUnlock], block_number: Union[raiden.utils.Address, raiden.transfer.state.TokenNetworkState, raiden.transfer.state_change.ReceiveUnlock], block_hash: Union[metrics_backend.utils.Address, int, raiden.utils.BlockNumber]) -> Union[list[ContractSendChannelCoopSettle], list]:
    if coop_settle_state.partner_signature_request is not None and coop_settle_state.partner_signature_confirmation is not None and (coop_settle_state.expiration >= block_number - channel_state.reveal_timeout):
        msg = 'CoopSettleState should be present in our state if we initiated it'
        assert channel_state.our_state.initiated_coop_settle is not None, msg
        send_coop_settle = ContractSendChannelCoopSettle(canonical_identifier=channel_state.canonical_identifier, our_total_withdraw=coop_settle_state.total_withdraw_initiator, partner_total_withdraw=coop_settle_state.total_withdraw_partner, expiration=coop_settle_state.expiration, signature_our_withdraw=coop_settle_state.partner_signature_confirmation, signature_partner_withdraw=coop_settle_state.partner_signature_request, triggered_by_block_hash=block_hash)
        channel_state.our_state.initiated_coop_settle.transaction = TransactionExecutionStatus(block_number, None, None)
        return [send_coop_settle]
    return []

@handle_state_transitions.register
def _handle_receive_withdraw_request(action: Union[raiden.utils.BlockNumber, raiden.utils.Address, random.Random], channel_state: Union[raiden.utils.BlockNumber, raiden.utils.Address, random.Random], block_number: Union[raiden.transfer.state.TokenNetworkState, raiden.utils.ChannelMap, raiden.utils.Dict], block_hash: Union[raiden.transfer.state.TokenNetworkState, raiden.utils.ChannelMap, raiden.utils.Dict], pseudo_random_generator: Union[raiden.utils.BlockNumber, raiden.utils.Address, random.Random]) -> Union[bool, dict, TransitionResult]:
    events = []

    def transition_result_invalid(success_or_error: Any) -> TransitionResult:
        error_msg = success_or_error.as_error_message
        assert error_msg, 'is_valid_withdraw_request should return error msg if not valid'
        invalid_withdraw_request = EventInvalidReceivedWithdrawRequest(attempted_withdraw=action.total_withdraw, reason=error_msg)
        return TransitionResult(channel_state, [invalid_withdraw_request])
    is_valid = is_valid_withdraw_request(channel_state=channel_state, withdraw_request=action)
    if not is_valid:
        return transition_result_invalid(is_valid)
    withdraw_state = PendingWithdrawState(total_withdraw=action.total_withdraw, nonce=action.nonce, expiration=action.expiration, recipient_metadata=action.sender_metadata)
    channel_state.partner_state.withdraws_pending[withdraw_state.total_withdraw] = withdraw_state
    channel_state.partner_state.nonce = action.nonce
    our_initiated_coop_settle = channel_state.our_state.initiated_coop_settle
    if our_initiated_coop_settle is not None or action.coop_settle:
        partner_max_total_withdraw = get_max_withdraw_amount(channel_state.partner_state, channel_state.our_state)
        if partner_max_total_withdraw != action.total_withdraw:
            return transition_result_invalid(SuccessOrError(f'Partner did not withdraw with maximum balance (should={partner_max_total_withdraw}).'))
        if get_number_of_pending_transfers(channel_state.partner_state) > 0:
            return transition_result_invalid(SuccessOrError('Partner has pending transfers.'))
        if our_initiated_coop_settle is not None:
            if our_initiated_coop_settle.expiration != action.expiration:
                return transition_result_invalid(SuccessOrError("Partner requested withdraw while we initiated a coop-settle: Partner's withdraw has differing expiration."))
            msg = "The expected total withdraw of the partner doesn't match the withdraw-request"
            assert our_initiated_coop_settle.total_withdraw_partner == action.total_withdraw, msg
            our_initiated_coop_settle.partner_signature_request = action.signature
            coop_settle_events = events_for_coop_settle(channel_state, our_initiated_coop_settle, block_number, block_hash)
            events.extend(coop_settle_events)
        else:
            our_max_total_withdraw = get_max_withdraw_amount(channel_state.our_state, channel_state.partner_state)
            if get_number_of_pending_transfers(channel_state.our_state) > 0:
                return transition_result_invalid(SuccessOrError('Partner initiated coop-settle, but we have pending transfers.'))
            partner_initiated_coop_settle = CoopSettleState(total_withdraw_initiator=action.total_withdraw, total_withdraw_partner=our_max_total_withdraw, expiration=action.expiration, partner_signature_request=action.signature)
            channel_state.partner_state.initiated_coop_settle = partner_initiated_coop_settle
            events = send_withdraw_request(channel_state=channel_state, expiration=partner_initiated_coop_settle.expiration, total_withdraw=our_max_total_withdraw, pseudo_random_generator=pseudo_random_generator, recipient_metadata=action.sender_metadata, coop_settle=False)
            events.extend(events)
    channel_state.our_state.nonce = get_next_nonce(channel_state.our_state)
    send_withdraw = SendWithdrawConfirmation(canonical_identifier=channel_state.canonical_identifier, recipient=channel_state.partner_state.address, recipient_metadata=action.sender_metadata, message_identifier=action.message_identifier, total_withdraw=action.total_withdraw, participant=channel_state.partner_state.address, nonce=channel_state.our_state.nonce, expiration=withdraw_state.expiration)
    events.append(send_withdraw)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_receive_withdraw_confirmation(action: Union[raiden.utils.BlockSpecification, raiden.utils.BlockIdentifier, int], channel_state: Union[raiden.utils.BlockSpecification, raiden.utils.BlockIdentifier, int], block_number: Union[metrics_backend.utils.Address, raiden.utils.BlockSpecification], block_hash: Union[metrics_backend.utils.Address, raiden.utils.BlockIdentifier], **kwargs) -> TransitionResult:
    is_valid = is_valid_withdraw_confirmation(channel_state=channel_state, received_withdraw=action)
    withdraw_state = channel_state.our_state.withdraws_pending.get(action.total_withdraw)
    recipient_metadata = None
    if withdraw_state is not None:
        recipient_metadata = withdraw_state.recipient_metadata
    if is_valid:
        channel_state.partner_state.nonce = action.nonce
        events = [SendProcessed(recipient=channel_state.partner_state.address, recipient_metadata=recipient_metadata, message_identifier=action.message_identifier, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)]
        our_initiated_coop_settle = channel_state.our_state.initiated_coop_settle
        partner_initiated_coop_settle = channel_state.partner_state.initiated_coop_settle
        if our_initiated_coop_settle is not None:
            assert partner_initiated_coop_settle is None, 'Only one party can initiate a coop settle'
            our_initiated_coop_settle.partner_signature_confirmation = action.signature
            coop_settle_events = events_for_coop_settle(channel_state, our_initiated_coop_settle, block_number, block_hash)
            events.extend(coop_settle_events)
        if our_initiated_coop_settle is None and partner_initiated_coop_settle is None:
            if action.expiration >= block_number - channel_state.reveal_timeout:
                withdraw_on_chain = ContractSendChannelWithdraw(canonical_identifier=action.canonical_identifier, total_withdraw=action.total_withdraw, partner_signature=action.signature, expiration=action.expiration, triggered_by_block_hash=block_hash)
                events.append(withdraw_on_chain)
    else:
        error_msg = is_valid.as_error_message
        assert error_msg, 'is_valid_withdraw_confirmation should return error msg if not valid'
        invalid_withdraw = EventInvalidReceivedWithdraw(attempted_withdraw=action.total_withdraw, reason=error_msg)
        events = [invalid_withdraw]
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_receive_withdraw_expired(action: Union[raiden.utils.Address, raiden.utils.TokenAmount, raiden.utils.FeeAmount], channel_state: Union[raiden.utils.Address, raiden.utils.BlockSpecification, raiden.utils.FeeAmount], block_number: Union[raiden.utils.BlockSpecification, raiden.utils.Address, raiden.utils.BlockIdentifier], **kwargs) -> TransitionResult:
    events = []
    withdraw_state = channel_state.partner_state.withdraws_pending.get(action.total_withdraw)
    if not withdraw_state:
        invalid_withdraw_expired_msg = f'Withdraw expired of {action.total_withdraw} did not correspond to previous withdraw request'
        return TransitionResult(channel_state, [EventInvalidReceivedWithdrawExpired(attempted_withdraw=action.total_withdraw, reason=invalid_withdraw_expired_msg)])
    is_valid = is_valid_withdraw_expired(channel_state=channel_state, state_change=action, withdraw_state=withdraw_state, block_number=block_number)
    if is_valid:
        del channel_state.partner_state.withdraws_pending[withdraw_state.total_withdraw]
        channel_state.partner_state.nonce = action.nonce
        coop_settle = channel_state.partner_state.initiated_coop_settle
        if coop_settle is not None:
            if coop_settle.total_withdraw_initiator == withdraw_state.total_withdraw and coop_settle.expiration == withdraw_state.expiration:
                channel_state.partner_state.initiated_coop_settle = None
        send_processed = SendProcessed(recipient=channel_state.partner_state.address, recipient_metadata=withdraw_state.recipient_metadata, message_identifier=action.message_identifier, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
        events = [send_processed]
    else:
        error_msg = is_valid.as_error_message
        assert error_msg, 'is_valid_withdraw_expired should return error msg if not valid'
        invalid_withdraw_expired = EventInvalidReceivedWithdrawExpired(attempted_withdraw=action.total_withdraw, reason=error_msg)
        events = [invalid_withdraw_expired]
    return TransitionResult(channel_state, events)

def handle_refundtransfer(received_transfer: Union[RaidenService, raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.transfer.mediated_transfer.state_change.TransferDescriptionWithSecretState], channel_state: Union[RaidenService, raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.transfer.mediated_transfer.state_change.TransferDescriptionWithSecretState], refund: Union[RaidenService, raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.transfer.mediated_transfer.state_change.TransferDescriptionWithSecretState]) -> tuple[typing.Union[list[EventInvalidReceivedTransferRefund],list[SendProcessed]]]:
    is_valid, msg, pending_locks = is_valid_refund(refund=refund, channel_state=channel_state, sender_state=channel_state.partner_state, receiver_state=channel_state.our_state, received_transfer=received_transfer)
    if is_valid:
        assert pending_locks, 'is_valid_refund should return pending locks if valid'
        channel_state.partner_state.balance_proof = refund.transfer.balance_proof
        channel_state.partner_state.nonce = refund.transfer.balance_proof.nonce
        channel_state.partner_state.pending_locks = pending_locks
        lock = refund.transfer.lock
        channel_state.partner_state.secrethashes_to_lockedlocks[lock.secrethash] = lock
        recipient_address = channel_state.partner_state.address
        recipient_metadata = get_address_metadata(recipient_address, received_transfer.route_states)
        send_processed = SendProcessed(recipient=refund.transfer.balance_proof.sender, recipient_metadata=recipient_metadata, message_identifier=refund.transfer.message_identifier, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
        events = [send_processed]
    else:
        assert msg, 'is_valid_refund should return error msg if not valid'
        invalid_refund = EventInvalidReceivedTransferRefund(payment_identifier=received_transfer.payment_identifier, reason=msg)
        events = [invalid_refund]
    return (is_valid, events, msg)

def handle_receive_lock_expired(channel_state: Union[raiden.transfer.state.TokenNetworkState, raiden.utils.ChannelMap, raiden.utils.BlockNumber], state_change: Union[raiden.transfer.state.TokenNetworkState, raiden.utils.ChannelMap, raiden.utils.BlockNumber], block_number: Union[raiden.transfer.state.TokenNetworkState, raiden.utils.ChannelMap, raiden.utils.BlockNumber], recipient_metadata: Union[None, raiden.utils.ChannelMap, raiden.utils.TokenAmount, raiden.transfer.state.ChainState]=None) -> TransitionResult:
    """Remove expired locks from channel states."""
    is_valid, msg, pending_locks = is_valid_lock_expired(state_change=state_change, channel_state=channel_state, sender_state=channel_state.partner_state, receiver_state=channel_state.our_state, block_number=block_number)
    events = []
    if is_valid:
        assert pending_locks, 'is_valid_lock_expired should return pending locks if valid'
        channel_state.partner_state.balance_proof = state_change.balance_proof
        channel_state.partner_state.nonce = state_change.balance_proof.nonce
        channel_state.partner_state.pending_locks = pending_locks
        _del_unclaimed_lock(channel_state.partner_state, state_change.secrethash)
        send_processed = SendProcessed(recipient=state_change.balance_proof.sender, recipient_metadata=recipient_metadata, message_identifier=state_change.message_identifier, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
        events = [send_processed]
    else:
        assert msg, 'is_valid_lock_expired should return error msg if not valid'
        invalid_lock_expired = EventInvalidReceivedLockExpired(secrethash=state_change.secrethash, reason=msg)
        events = [invalid_lock_expired]
    return TransitionResult(channel_state, events)

def handle_receive_lockedtransfer(channel_state: Union[raiden.transfer.mediated_transfer.state.InitiatorPaymentState, str, raiden.transfer.mediated_transfer.state_change.ReceiveTransferRefundCancelRoute], mediated_transfer: Union[raiden.transfer.mediated_transfer.state.InitiatorPaymentState, str, raiden.transfer.mediated_transfer.state_change.ReceiveTransferRefundCancelRoute], recipient_metadata: Union[None, raiden.transfer.state.ChainState, bytes, raiden.utils.BlockNumber]=None) -> tuple[typing.Union[list[EventInvalidReceivedLockedTransfer],list[SendProcessed]]]:
    """Register the latest known transfer.

    The receiver needs to use this method to update the container with a
    _valid_ transfer, otherwise the locksroot will not contain the pending
    transfer. The receiver needs to ensure that the locksroot has the
    secrethash included, otherwise it won't be able to claim it.
    """
    is_valid, msg, pending_locks = is_valid_lockedtransfer(mediated_transfer, channel_state, channel_state.partner_state, channel_state.our_state)
    if is_valid:
        assert pending_locks, 'is_valid_lock_expired should return pending locks if valid'
        channel_state.partner_state.balance_proof = mediated_transfer.balance_proof
        channel_state.partner_state.nonce = mediated_transfer.balance_proof.nonce
        channel_state.partner_state.pending_locks = pending_locks
        lock = mediated_transfer.lock
        channel_state.partner_state.secrethashes_to_lockedlocks[lock.secrethash] = lock
        send_processed = SendProcessed(recipient=mediated_transfer.balance_proof.sender, recipient_metadata=recipient_metadata, message_identifier=mediated_transfer.message_identifier, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
        events = [send_processed]
    else:
        assert msg, 'is_valid_lock_expired should return error msg if not valid'
        invalid_locked = EventInvalidReceivedLockedTransfer(payment_identifier=mediated_transfer.payment_identifier, reason=msg)
        events = [invalid_locked]
    return (is_valid, events, msg)

def handle_unlock(channel_state: raiden.transfer.state.ChainState, unlock: raiden.transfer.state.ChainState, recipient_metadata: Union[None, int, raiden.blockchain.events.DecodedEvent, dict]=None) -> tuple[typing.Union[list[EventInvalidReceivedUnlock],list[SendProcessed]]]:
    is_valid, msg, unlocked_pending_locks = is_valid_unlock(unlock, channel_state, channel_state.partner_state)
    if is_valid:
        assert unlocked_pending_locks is not None, 'is_valid_unlock should return pending locks if valid'
        channel_state.partner_state.balance_proof = unlock.balance_proof
        channel_state.partner_state.nonce = unlock.balance_proof.nonce
        channel_state.partner_state.pending_locks = unlocked_pending_locks
        _del_lock(channel_state.partner_state, unlock.secrethash)
        send_processed = SendProcessed(recipient=unlock.balance_proof.sender, recipient_metadata=recipient_metadata, message_identifier=unlock.message_identifier, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
        events = [send_processed]
    else:
        assert msg, 'is_valid_unlock should return error msg if not valid'
        invalid_unlock = EventInvalidReceivedUnlock(secrethash=unlock.secrethash, reason=msg)
        events = [invalid_unlock]
    return (is_valid, events, msg)

@handle_state_transitions.register
def _handle_block(action: Union[raiden.utils.ChannelMap, raiden.utils.BlockSpecification, raiden.transfer.state.TokenNetworkState], channel_state: Union[raiden.utils.ChannelMap, raiden.utils.Address, raiden.utils.BlockNumber], block_number: Union[raiden.utils.Address, RaidenService, raiden.utils.BlockIdentifier], pseudo_random_generator: Union[raiden.utils.Address, raiden.utils.TokenAmount, raiden.utils.FeeAmount], **kwargs) -> TransitionResult:
    assert action.block_number == block_number, 'Block number mismatch'
    events = []
    if get_status(channel_state) == ChannelState.STATE_OPENED:
        expired_withdraws = events_for_expired_withdraws(channel_state=channel_state, block_number=block_number, pseudo_random_generator=pseudo_random_generator)
        events.extend(expired_withdraws)
    if get_status(channel_state) == ChannelState.STATE_CLOSED:
        msg = 'channel get_status is STATE_CLOSED, but close_transaction is not set'
        assert channel_state.close_transaction, msg
        msg = 'channel get_status is STATE_CLOSED, but close_transaction block number is missing'
        assert channel_state.close_transaction.finished_block_number, msg
        closed_block_number = channel_state.close_transaction.finished_block_number
        settlement_end = closed_block_number + channel_state.settle_timeout
        if action.block_number > settlement_end:
            channel_state.settle_transaction = TransactionExecutionStatus(action.block_number, None, None)
            event = ContractSendChannelSettle(canonical_identifier=channel_state.canonical_identifier, triggered_by_block_hash=action.block_hash)
            events.append(event)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_channel_closed(action: Union[raiden.transfer.state.NettingChannelState, int], channel_state: Union[raiden.transfer.state.NettingChannelState, int], **kwargs) -> TransitionResult:
    events = []
    just_closed = action.channel_identifier == channel_state.identifier and get_status(channel_state) in CHANNEL_STATES_PRIOR_TO_CLOSED
    if just_closed:
        set_closed(channel_state, action.block_number)
        balance_proof = channel_state.partner_state.balance_proof
        call_update = action.transaction_from != channel_state.our_state.address and balance_proof is not None and (channel_state.update_transaction is None)
        if call_update:
            expiration = BlockExpiration(action.block_number + channel_state.settle_timeout)
            assert isinstance(balance_proof, BalanceProofSignedState), MYPY_ANNOTATION
            update = ContractSendChannelUpdateTransfer(expiration=expiration, balance_proof=balance_proof, triggered_by_block_hash=action.block_hash)
            channel_state.update_transaction = TransactionExecutionStatus(started_block_number=action.block_number, finished_block_number=None, result=None)
            events.append(update)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_channel_updated_transfer(action: Union[raiden.utils.BlockSpecification, raiden.utils.BlockIdentifier, int], channel_state: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification, raiden.utils.TokenAddress], block_number: Union[raiden.utils.Address, raiden.utils.BlockNumber, raiden.utils.FeeAmount], **kwargs) -> TransitionResult:
    if action.channel_identifier == channel_state.identifier:
        channel_state.update_transaction = TransactionExecutionStatus(started_block_number=None, finished_block_number=block_number, result=TransactionExecutionStatus.SUCCESS)
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_channel_settled(action: Union[raiden.transfer.state.NettingChannelState, str, raiden.utils.Address], channel_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.state.ChainState, raiden.transfer.mediated_transfer.state.InitiatorPaymentState], **kwargs) -> TransitionResult:
    events = []
    if action.channel_identifier == channel_state.identifier:
        set_settled(channel_state, action.block_number)
        our_locksroot = action.our_onchain_locksroot
        partner_locksroot = action.partner_onchain_locksroot
        should_clear_channel = our_locksroot == LOCKSROOT_OF_NO_LOCKS and partner_locksroot == LOCKSROOT_OF_NO_LOCKS
        is_coop_settle = False
        initiator_lock_check = action.our_onchain_locksroot == LOCKSROOT_OF_NO_LOCKS
        partner_lock_check = action.partner_onchain_locksroot == LOCKSROOT_OF_NO_LOCKS
        if channel_state.our_state.initiated_coop_settle:
            coop_settle = channel_state.our_state.initiated_coop_settle
            initiator_transfer_check = TokenAmount(coop_settle.total_withdraw_initiator) == action.our_transferred_amount
            partner_transfer_check = TokenAmount(coop_settle.total_withdraw_partner) == action.partner_transferred_amount
            initiator_checks = initiator_transfer_check and initiator_lock_check
            partner_checks = partner_transfer_check and partner_lock_check
            if initiator_checks and partner_checks:
                set_coop_settled(channel_state.our_state, action.block_number)
                is_coop_settle = True
        if channel_state.partner_state.initiated_coop_settle:
            coop_settle = channel_state.partner_state.initiated_coop_settle
            partner_transfer_check = TokenAmount(coop_settle.total_withdraw_initiator) == action.partner_transferred_amount
            initiator_transfer_check = TokenAmount(coop_settle.total_withdraw_partner) == action.our_transferred_amount
            initiator_checks = initiator_transfer_check and initiator_lock_check
            partner_checks = partner_transfer_check and partner_lock_check
            if initiator_checks and partner_checks:
                set_coop_settled(channel_state.partner_state, action.block_number)
                is_coop_settle = True
        if is_coop_settle:
            channel_state.partner_state.onchain_total_withdraw = WithdrawAmount(action.partner_transferred_amount)
            channel_state.our_state.onchain_total_withdraw = WithdrawAmount(action.our_transferred_amount)
        if should_clear_channel:
            return TransitionResult(None, events)
        channel_state.our_state.onchain_locksroot = our_locksroot
        channel_state.partner_state.onchain_locksroot = partner_locksroot
        onchain_unlock = ContractSendChannelBatchUnlock(canonical_identifier=channel_state.canonical_identifier, sender=channel_state.partner_state.address, triggered_by_block_hash=action.block_hash)
        events.append(onchain_unlock)
    return TransitionResult(channel_state, events)

def update_fee_schedule_after_balance_change(channel_state: Union[raiden.blockchain.events.DecodedEvent, raiden.transfer.state.ChainState, raiden.transfer.mediated_transfer.state.InitiatorPaymentState], fee_config: Union[raiden.transfer.state.ChainState, raiden.blockchain.events.DecodedEvent, raiden.transfer.mediated_transfer.state.InitiatorPaymentState]) -> list:
    proportional_imbalance_fee = fee_config.get_proportional_imbalance_fee(channel_state.token_address)
    imbalance_penalty = calculate_imbalance_fees(channel_capacity=get_capacity(channel_state), proportional_imbalance_fee=proportional_imbalance_fee)
    channel_state.fee_schedule = FeeScheduleState(cap_fees=channel_state.fee_schedule.cap_fees, flat=channel_state.fee_schedule.flat, proportional=channel_state.fee_schedule.proportional, imbalance_penalty=imbalance_penalty)
    return []

@handle_state_transitions.register
def _handle_channel_deposit(action: Union[raiden.transfer.state.NettingChannelState, int, raiden.transfer.mediated_transfer.state.InitiatorPaymentState], channel_state: Union[raiden.transfer.state.NettingChannelState, int, raiden.transfer.mediated_transfer.state.InitiatorPaymentState], **kwargs) -> TransitionResult:
    participant_address = action.deposit_transaction.participant_address
    contract_balance = Balance(action.deposit_transaction.contract_balance)
    if participant_address == channel_state.our_state.address:
        update_contract_balance(channel_state.our_state, contract_balance)
    elif participant_address == channel_state.partner_state.address:
        update_contract_balance(channel_state.partner_state, contract_balance)
    update_fee_schedule_after_balance_change(channel_state, action.fee_config)
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_channel_withdraw(action: Union[raiden.transfer.state.NettingChannelState, int], channel_state: Union[raiden.transfer.state.NettingChannelState, int, raiden.transfer.mediated_transfer.state.InitiatorPaymentState], **kwargs) -> TransitionResult:
    """An on-chain total withdraw took place which means that we have to keep
    track of this not to go lower than the on-chain value. The value is set to
    onchain_total_withdraw and the corresponding withdraw_state is cleared.
    """
    participants = (channel_state.our_state.address, channel_state.partner_state.address)
    if action.participant not in participants:
        return TransitionResult(channel_state, [])
    if action.participant == channel_state.our_state.address:
        end_state = channel_state.our_state
    else:
        end_state = channel_state.partner_state
    withdraw_state = end_state.withdraws_pending.get(action.total_withdraw)
    if withdraw_state:
        del end_state.withdraws_pending[action.total_withdraw]
    end_state.onchain_total_withdraw = action.total_withdraw
    update_fee_schedule_after_balance_change(channel_state, action.fee_config)
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_channel_batch_unlock(action: Union[raiden.transfer.state.NettingChannelState, int, str], channel_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.mediated_transfer.state.InitiatorPaymentState, int], **kwargs) -> TransitionResult:
    events = []
    new_channel_state = channel_state
    if get_status(channel_state) == ChannelState.STATE_SETTLED:
        our_state = channel_state.our_state
        partner_state = channel_state.partner_state
        if action.sender == our_state.address:
            our_state.onchain_locksroot = Locksroot(LOCKSROOT_OF_NO_LOCKS)
        elif action.sender == partner_state.address:
            partner_state.onchain_locksroot = Locksroot(LOCKSROOT_OF_NO_LOCKS)
        no_unlock_left_to_do = our_state.onchain_locksroot == Locksroot(LOCKSROOT_OF_NO_LOCKS) and partner_state.onchain_locksroot == Locksroot(LOCKSROOT_OF_NO_LOCKS)
        if no_unlock_left_to_do:
            new_channel_state = None
    return TransitionResult(new_channel_state, events)

def sanity_check(channel_state: Union[raiden.transfer.state.NettingChannelState, int, raiden.transfer.mediated_transfer.state.InitiatorPaymentState]) -> None:
    """Some of the checks below are tautologies for the current version of the
    codebase. However they are kept in there to check the constraints if/when
    the code changes.
    """
    partner_state = channel_state.partner_state
    our_state = channel_state.our_state
    previous = WithdrawAmount(0)
    coop_settle = channel_state.our_state.initiated_coop_settle is not None or channel_state.partner_state.initiated_coop_settle is not None
    for total_withdraw, withdraw_state in our_state.withdraws_pending.items():
        if not coop_settle:
            assert withdraw_state.total_withdraw > previous, 'total_withdraw must be ordered'
        assert total_withdraw == withdraw_state.total_withdraw, 'Total withdraw mismatch'
        previous = withdraw_state.total_withdraw
    our_balance = get_balance(our_state, partner_state)
    partner_balance = get_balance(partner_state, our_state)
    msg = 'The balance can never be negative, that would be equivalent to a loan or a double spend.'
    assert our_balance >= 0, msg
    assert partner_balance >= 0, msg
    channel_capacity = get_capacity(channel_state)
    msg = 'The whole deposit of the channel has to be accounted for.'
    assert our_balance + partner_balance == channel_capacity, msg
    our_locked = get_amount_locked(our_state)
    partner_locked = get_amount_locked(partner_state)
    our_bp_locksroot, _, _, our_bp_locked_amount = get_current_balanceproof(our_state)
    partner_bp_locksroot, _, _, partner_bp_locked_amount = get_current_balanceproof(partner_state)
    msg = "The sum of the lock's amounts, and the value of the balance proof locked_amount must be equal, otherwise settle will not reserve the proper amount of tokens."
    assert partner_locked == partner_bp_locked_amount, msg
    assert our_locked == our_bp_locked_amount, msg
    our_distributable = get_distributable(our_state, partner_state)
    partner_distributable = get_distributable(partner_state, our_state)
    assert our_distributable + our_locked <= our_balance, 'distributable + locked must not exceed balance (own)'
    assert partner_distributable + partner_locked <= partner_balance, 'distributable + locked must not exceed balance (partner)'
    our_locksroot = compute_locksroot(our_state.pending_locks)
    partner_locksroot = compute_locksroot(partner_state.pending_locks)
    msg = 'The balance proof locks root must match the existing locks, otherwise it is not possible to prove on-chain that a given lock was pending.'
    assert our_locksroot == our_bp_locksroot, msg
    assert partner_locksroot == partner_bp_locksroot, msg
    msg = 'The lock mappings and the pending locks must be synchronized, otherwise there is a bug'
    for lock in partner_state.secrethashes_to_lockedlocks.values():
        assert lock.encoded in partner_state.pending_locks.locks, msg
    for partial_unlock in partner_state.secrethashes_to_unlockedlocks.values():
        assert partial_unlock.encoded in partner_state.pending_locks.locks, msg
    for partial_unlock in partner_state.secrethashes_to_onchain_unlockedlocks.values():
        assert partial_unlock.encoded in partner_state.pending_locks.locks, msg
    for lock in our_state.secrethashes_to_lockedlocks.values():
        assert lock.encoded in our_state.pending_locks.locks, msg
    for partial_unlock in our_state.secrethashes_to_unlockedlocks.values():
        assert partial_unlock.encoded in our_state.pending_locks.locks, msg
    for partial_unlock in our_state.secrethashes_to_onchain_unlockedlocks.values():
        assert partial_unlock.encoded in our_state.pending_locks.locks, msg

def state_transition(channel_state: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState], state_change: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState], block_number: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState], block_hash: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState], pseudo_random_generator: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState]) -> Union[raiden.transfer.architecture.TransitionResult, int]:
    iteration = handle_state_transitions(state_change, channel_state=channel_state, block_number=block_number, block_hash=block_hash, pseudo_random_generator=pseudo_random_generator)
    if iteration.new_state is not None:
        sanity_check(iteration.new_state)
    return iteration