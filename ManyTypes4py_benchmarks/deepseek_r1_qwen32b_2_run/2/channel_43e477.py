import random
from enum import Enum
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union
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
from raiden.transfer.identifiers import (
    CANONICAL_IDENTIFIER_UNORDERED_QUEUE,
    CanonicalIdentifier,
)
from raiden.transfer.mediated_transfer.events import SendLockedTransfer, SendLockExpired, SendUnlock
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState, calculate_imbalance_fees
from raiden.transfer.mediated_transfer.state import (
    LockedTransferSignedState,
    LockedTransferUnsignedState,
)
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
    Dict,
    EncodedData,
    FeeAmount,
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
    pass

def get_sender_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    pass

def get_receiver_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    pass

def is_channel_usable_for_mediation(channel_state: NettingChannelState, transfer_amount: TokenAmount, lock_timeout: BlockTimeout) -> bool:
    pass

def is_channel_usable_for_new_transfer(channel_state: NettingChannelState, transfer_amount: TokenAmount, lock_timeout: BlockTimeout) -> ChannelUsability:
    pass

def is_lock_pending(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    pass

def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    pass

def is_lock_expired(end_state: NettingChannelEndState, lock: HashTimeLockState, block_number: BlockNumber, lock_expiration_threshold: BlockExpiration) -> SuccessOrError:
    pass

def is_transfer_expired(transfer: LockedTransferSignedState, affected_channel: NettingChannelState, block_number: BlockNumber) -> bool:
    pass

def is_withdraw_expired(block_number: BlockNumber, expiration_threshold: BlockExpiration) -> bool:
    pass

def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    pass

def is_secret_known_offchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    pass

def is_secret_known_onchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    pass

def is_valid_channel_total_withdraw(channel_total_withdraw: TokenAmount) -> bool:
    pass

def is_valid_withdraw(withdraw_request: SendWithdrawRequest) -> SuccessOrError:
    pass

def get_secret(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[Secret]:
    pass

def is_balance_proof_safe_for_onchain_operations(balance_proof: BalanceProofUnsignedState) -> bool:
    pass

def is_valid_amount(end_state: NettingChannelEndState, amount: TokenAmount) -> bool:
    pass

def is_valid_signature(data: EncodedData, signature: Signature, sender_address: Address) -> SuccessOrError:
    pass

def is_valid_balanceproof_signature(balance_proof: BalanceProofUnsignedState, sender_address: Address) -> SuccessOrError:
    pass

def is_balance_proof_usable_onchain(received_balance_proof: BalanceProofUnsignedState, channel_state: NettingChannelState, sender_state: NettingChannelEndState) -> SuccessOrError:
    pass

def is_valid_lockedtransfer(transfer_state: LockedTransferUnsignedState, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState) -> bool:
    pass

def is_valid_lock_expired(state_change: ReceiveLockExpired, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, block_number: BlockNumber) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    pass

def valid_lockedtransfer_check(channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, message_name: str, received_balance_proof: BalanceProofUnsignedState, lock: HashTimeLockState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    pass

def refund_transfer_matches_transfer(refund_transfer: LockedTransferUnsignedState, transfer: LockedTransferUnsignedState) -> bool:
    pass

def is_valid_refund(refund: ReceiveTransferRefund, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, received_transfer: LockedTransferUnsignedState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    pass

def is_valid_unlock(unlock: ReceiveUnlock, channel_state: NettingChannelState, sender_state: NettingChannelEndState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    pass

def is_valid_total_withdraw(channel_state: NettingChannelState, our_total_withdraw: TokenAmount, allow_zero: bool = False) -> SuccessOrError:
    pass

def is_valid_action_coopsettle(channel_state: NettingChannelState, coop_settle: CoopSettleState, total_withdraw: TokenAmount) -> SuccessOrError:
    pass

def is_valid_withdraw_request(channel_state: NettingChannelState, withdraw_request: SendWithdrawRequest) -> SuccessOrError:
    pass

def is_valid_withdraw_confirmation(channel_state: NettingChannelState, received_withdraw: ReceiveWithdrawConfirmation) -> SuccessOrError:
    pass

def is_valid_withdraw_expired(channel_state: NettingChannelState, state_change: ReceiveWithdrawExpired, withdraw_state: ExpiredWithdrawState, block_number: BlockNumber) -> SuccessOrError:
    pass

def get_amount_unclaimed_onchain(end_state: NettingChannelEndState) -> TokenAmount:
    pass

def get_amount_locked(end_state: NettingChannelEndState) -> LockedAmount:
    pass

def get_capacity(channel_state: NettingChannelState) -> TokenAmount:
    pass

def get_balance(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> Balance:
    pass

def get_max_withdraw_amount(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> WithdrawAmount:
    pass

def get_current_balanceproof(end_state: NettingChannelEndState) -> Tuple[Locksroot, Nonce, TokenAmount, LockedAmount]:
    pass

def get_current_nonce(end_state: NettingChannelEndState) -> Nonce:
    pass

def get_distributable(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> TokenAmount:
    pass

def get_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[HashTimeLockState]:
    pass

def lock_exists_in_either_channel_side(channel_state: NettingChannelState, secrethash: SecretHash) -> bool:
    pass

def get_next_nonce(end_state: NettingChannelEndState) -> Nonce:
    pass

def get_number_of_pending_transfers(channel_end_state: NettingChannelEndState) -> int:
    pass

def get_status(channel_state: NettingChannelState) -> ChannelState:
    pass

def _del_unclaimed_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None:
    pass

def _del_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None:
    pass

def set_closed(channel_state: NettingChannelState, block_number: BlockNumber) -> None:
    pass

def set_settled(channel_state: NettingChannelState, block_number: BlockNumber) -> None:
    pass

def set_coop_settled(channel_end_state: NettingChannelEndState, block_number: BlockNumber) -> None:
    pass

def update_contract_balance(end_state: NettingChannelEndState, contract_balance: Balance) -> None:
    pass

def compute_locks_with(locks: PendingLocksState, lock: HashTimeLockState) -> Optional[PendingLocksState]:
    pass

def compute_locks_without(locks: PendingLocksState, lock_encoded: EncodedData) -> Optional[PendingLocksState]:
    pass

def compute_locksroot(locks: PendingLocksState) -> Locksroot:
    pass

def create_sendlockedtransfer(
    channel_state: NettingChannelState,
    initiator: InitiatorAddress,
    target: TargetAddress,
    amount: TokenAmount,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    expiration: BlockExpiration,
    secret: Secret,
    secrethash: SecretHash,
    route_states: List[RouteState],
    recipient_metadata: Optional[AddressMetadata] = None,
    previous_metadata: Optional[AddressMetadata] = None,
) -> Tuple[SendLockedTransfer, PendingLocksState]:
    pass

def create_unlock(
    channel_state: NettingChannelState,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    secret: Secret,
    lock: HashTimeLockState,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> Tuple[SendUnlock, PendingLocksState]:
    pass

def send_lockedtransfer(
    channel_state: NettingChannelState,
    initiator: InitiatorAddress,
    target: TargetAddress,
    amount: TokenAmount,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    expiration: BlockExpiration,
    secret: Secret,
    secrethash: SecretHash,
    route_states: List[RouteState],
    recipient_metadata: Optional[AddressMetadata] = None,
    previous_metadata: Optional[AddressMetadata] = None,
) -> SendLockedTransfer:
    pass

def send_unlock(
    channel_state: NettingChannelState,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    secret: Secret,
    secrethash: SecretHash,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> SendUnlock:
    pass

def events_for_close(channel_state: NettingChannelState, block_number: BlockNumber, block_hash: BlockHash) -> List[Event]:
    pass

def send_withdraw_request(
    channel_state: NettingChannelState,
    total_withdraw: TokenAmount,
    expiration: BlockExpiration,
    pseudo_random_generator: random.Random,
    recipient_metadata: Optional[AddressMetadata] = None,
    coop_settle: bool = False,
) -> List[Event]:
    pass

def create_sendexpiredlock(
    sender_end_state: NettingChannelEndState,
    locked_lock: HashTimeLockState,
    pseudo_random_generator: random.Random,
    chain_id: ChainID,
    token_network_address: TokenNetworkAddress,
    channel_identifier: ChannelID,
    recipient: Address,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> Tuple[Optional[SendLockExpired], Optional[PendingLocksState]]:
    pass

def send_lock_expired(
    channel_state: NettingChannelState,
    locked_lock: HashTimeLockState,
    pseudo_random_generator: random.Random,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> List[Event]:
    pass

def events_for_expired_withdraws(
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    pseudo_random_generator: random.Random,
) -> List[Event]:
    pass

def register_secret_endstate(end_state: NettingChannelEndState, secret: Secret, secrethash: SecretHash) -> None:
    pass

def register_onchain_secret_endstate(
    end_state: NettingChannelEndState,
    secret: Secret,
    secrethash: SecretHash,
    secret_reveal_block_number: BlockNumber,
    delete_lock: bool = True,
) -> None:
    pass

def register_offchain_secret(channel_state: NettingChannelState, secret: Secret, secrethash: SecretHash) -> None:
    pass

def register_onchain_secret(
    channel_state: NettingChannelState,
    secret: Secret,
    secrethash: SecretHash,
    secret_reveal_block_number: BlockNumber,
    delete_lock: bool = True,
) -> None:
    pass

@singledispatch
def handle_state_transitions(
    action: StateChange,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_action_close(
    action: ActionChannelClose,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_action_coop_settle(
    action: ActionChannelCoopSettle,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_action_withdraw(
    action: ActionChannelWithdraw,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_action_set_reveal_timeout(
    action: ActionChannelSetRevealTimeout,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    pass

def events_for_coop_settle(
    channel_state: NettingChannelState,
    coop_settle_state: CoopSettleState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[Event]:
    pass

@handle_state_transitions.register
def _handle_receive_withdraw_request(
    action: SendWithdrawRequest,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_receive_withdraw_confirmation(
    action: ReceiveWithdrawConfirmation,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_receive_withdraw_expired(
    action: ReceiveWithdrawExpired,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    pass

def handle_refundtransfer(
    received_transfer: LockedTransferUnsignedState,
    channel_state: NettingChannelState,
    refund: ReceiveTransferRefund,
) -> Tuple[bool, List[Event], Optional[str]]:
    pass

def handle_receive_lock_expired(
    channel_state: NettingChannelState,
    state_change: ReceiveLockExpired,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> TransitionResult:
    pass

def handle_receive_lockedtransfer(
    channel_state: NettingChannelState,
    mediated_transfer: LockedTransferUnsignedState,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> Tuple[bool, List[Event], Optional[str]]:
    pass

def handle_unlock(
    channel_state: NettingChannelState,
    unlock: ReceiveUnlock,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> Tuple[bool, List[Event], Optional[str]]:
    pass

@handle_state_transitions.register
def _handle_block(
    action: Block,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    pseudo_random_generator: random.Random,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_channel_closed(
    action: ContractReceiveChannelClosed,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_channel_updated_transfer(
    action: ContractReceiveUpdateTransfer,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_channel_settled(
    action: ContractReceiveChannelSettled,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_channel_deposit(
    action: ContractReceiveChannelDeposit,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_channel_withdraw(
    action: ContractReceiveChannelWithdraw,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    pass

@handle_state_transitions.register
def _handle_channel_batch_unlock(
    action: ContractReceiveChannelBatchUnlock,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    pass

def update_fee_schedule_after_balance_change(
    channel_state: NettingChannelState,
    fee_config: MediationFeeConfig,
) -> List[Event]:
    pass

def sanity_check(channel_state: NettingChannelState) -> None:
    pass

def state_transition(
    channel_state: NettingChannelState,
    state_change: StateChange,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    pass