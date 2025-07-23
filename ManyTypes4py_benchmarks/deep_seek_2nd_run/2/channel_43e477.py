import random
from enum import Enum
from functools import singledispatch
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
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
from raiden.transfer.mediated_transfer.events import SendLockedTransfer, SendLockExpired, SendUnlock
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState, calculate_imbalance_fees
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState, LockedTransferUnsignedState
from raiden.transfer.mediated_transfer.state_change import ReceiveLockExpired, ReceiveTransferRefund
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
    from raiden.raiden_service import RaidenService

T = TypeVar('T')

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
    if lock_timeout:
        expiration = block_number + lock_timeout
    else:
        expiration = block_number + reveal_timeout * 2
    return BlockExpiration(expiration - 1)

def get_sender_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS * 2)

def get_receiver_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    return BlockExpiration(expiration + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS)

def is_channel_usable_for_mediation(channel_state: ChannelState, transfer_amount: PaymentAmount, lock_timeout: BlockTimeout) -> bool:
    channel_usable = is_channel_usable_for_new_transfer(channel_state, transfer_amount, lock_timeout)
    return channel_usable is ChannelUsability.USABLE

def is_channel_usable_for_new_transfer(channel_state: ChannelState, transfer_amount: PaymentAmount, lock_timeout: Optional[BlockTimeout]) -> ChannelUsability:
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
    return secrethash in end_state.secrethashes_to_lockedlocks or secrethash in end_state.secrethashes_to_unlockedlocks or secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_lockedlocks

def is_lock_expired(end_state: NettingChannelEndState, lock: HashTimeLockState, block_number: BlockNumber, lock_expiration_threshold: BlockExpiration) -> SuccessOrError:
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
    return block_number >= expiration_threshold

def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_unlockedlocks or secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_secret_known_offchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_unlockedlocks

def is_secret_known_onchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_valid_channel_total_withdraw(channel_total_withdraw: TokenAmount) -> bool:
    return channel_total_withdraw <= UINT256_MAX

def is_valid_withdraw(withdraw_request: ReceiveWithdrawRequest) -> bool:
    packed = pack_withdraw(canonical_identifier=withdraw_request.canonical_identifier, participant=withdraw_request.participant, total_withdraw=withdraw_request.total_withdraw, expiration_block=withdraw_request.expiration)
    return is_valid_signature(data=packed, signature=withdraw_request.signature, sender_address=withdraw_request.sender)

def is_valid_action_coopsettle(channel_state: ChannelState, coop_settle: ActionChannelCoopSettle, total_withdraw: WithdrawAmount) -> SuccessOrError:
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

def is_valid_withdraw_request(channel_state: ChannelState, withdraw_request: ReceiveWithdrawRequest) -> SuccessOrError:
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

def is_valid_withdraw_confirmation(channel_state: ChannelState, received_withdraw: ReceiveWithdrawConfirmation) -> SuccessOrError:
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

def is_valid_withdraw_expired(channel_state: ChannelState, state_change: ReceiveWithdrawExpired, withdraw_state: PendingWithdrawState, block_number: BlockNumber) -> SuccessOrError:
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

def get_amount_unclaimed_onchain(end_state: NettingChannelEndState) -> TokenAmount:
    return TokenAmount(sum((unlock.lock.amount for unlock in end_state.secrethashes_to_onchain_unlockedlocks.values())))

def get_amount_locked(end_state: NettingChannelEndState) -> LockedAmount:
    total_pending = sum((lock.amount for lock in end_state.secrethashes_to_lockedlocks.values()))
    total_unclaimed = sum((unlock.lock.amount for unlock in end_state.secrethashes_to_unlockedlocks.values()))
    total_unclaimed_onchain = get_amount_unclaimed_onchain(end_state)
    result = total_pending + total_unclaimed + total_unclaimed_onchain
    return LockedAmount(result)

def get_capacity(channel_state: ChannelState) -> TokenAmount:
    return TokenAmount(channel_state.our_total_deposit - channel_state.our_total_withdraw + channel_state.partner_total_deposit - channel_state.partner_total_withdraw)

def get_balance(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> Balance:
    return _get_balance(sender, receiver, subtract_withdraws=True)

def _get_balance(sender: NettingChannelEndState, receiver: NettingChannelEndState, subtract_withdraws: bool = True) -> Balance:
    sender_transferred_amount = 0
    receiver_transferred_amount = 0
    if sender.balance_proof:
        sender_transferred_amount = sender.balance_proof.transferred_amount
    if receiver.balance_proof:
        receiver_transferred_amount = receiver.balance_proof.transferred_amount
    max