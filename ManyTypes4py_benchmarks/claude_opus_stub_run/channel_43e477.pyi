import random
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from raiden.settings import MediationFeeConfig
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
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import (
    SendLockedTransfer,
    SendLockExpired,
    SendUnlock,
)
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState
from raiden.transfer.mediated_transfer.state import (
    LockedTransferSignedState,
    LockedTransferUnsignedState,
)
from raiden.transfer.mediated_transfer.state_change import (
    ReceiveLockExpired,
    ReceiveTransferRefund,
)
from raiden.transfer.state import (
    BalanceProofSignedState,
    BalanceProofUnsignedState,
    ChannelState,
    CoopSettleState,
    HashTimeLockState,
    NettingChannelEndState,
    NettingChannelState,
    PendingLocksState,
    PendingWithdrawState,
    RouteState,
    TransactionExecutionStatus,
    UnlockPartialProofState,
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
from raiden.utils.typing import (
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
)

if TYPE_CHECKING:
    from raiden.raiden_service import RaidenService

PendingLocksStateOrError = Tuple[bool, Optional[str], Optional[PendingLocksState]]
EventsOrError = Tuple[bool, List[Event], Optional[str]]
BalanceProofData = Tuple[Locksroot, Nonce, TokenAmount, LockedAmount]
SendUnlockAndPendingLocksState = Tuple[SendUnlock, PendingLocksState]

class UnlockGain(NamedTuple): ...

class ChannelUsability(Enum):
    USABLE = True
    NOT_OPENED = 'channel is not open'
    INVALID_SETTLE_TIMEOUT = 'channel settle timeout is too low'
    CHANNEL_REACHED_PENDING_LIMIT = 'channel reached limit of pending transfers'
    CHANNEL_DOESNT_HAVE_ENOUGH_DISTRIBUTABLE = "channel doesn't have enough distributable tokens"
    CHANNEL_BALANCE_PROOF_WOULD_OVERFLOW = 'channel balance proof would overflow'
    LOCKTIMEOUT_MISMATCH = 'the lock timeout can not be used with the channel'

def get_safe_initial_expiration(
    block_number: BlockNumber,
    reveal_timeout: BlockTimeout,
    lock_timeout: Optional[BlockTimeout] = ...,
) -> BlockExpiration: ...

def get_sender_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration: ...

def get_receiver_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration: ...

def is_channel_usable_for_mediation(
    channel_state: NettingChannelState,
    transfer_amount: PaymentWithFeeAmount,
    lock_timeout: Optional[BlockTimeout],
) -> bool: ...

def is_channel_usable_for_new_transfer(
    channel_state: NettingChannelState,
    transfer_amount: PaymentWithFeeAmount,
    lock_timeout: Optional[BlockTimeout],
) -> ChannelUsability: ...

def is_lock_pending(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool: ...

def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool: ...

def is_lock_expired(
    end_state: NettingChannelEndState,
    lock: HashTimeLockState,
    block_number: BlockNumber,
    lock_expiration_threshold: BlockExpiration,
) -> SuccessOrError: ...

def is_transfer_expired(
    transfer: LockedTransferSignedState,
    affected_channel: NettingChannelState,
    block_number: BlockNumber,
) -> bool: ...

def is_withdraw_expired(
    block_number: BlockNumber,
    expiration_threshold: BlockExpiration,
) -> bool: ...

def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool: ...

def is_secret_known_offchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool: ...

def is_secret_known_onchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool: ...

def is_valid_channel_total_withdraw(channel_total_withdraw: TokenAmount) -> bool: ...

def is_valid_withdraw(withdraw_request: Any) -> SuccessOrError: ...

def get_secret(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[Secret]: ...

def is_balance_proof_safe_for_onchain_operations(
    balance_proof: Union[BalanceProofSignedState, BalanceProofUnsignedState],
) -> bool: ...

def is_valid_amount(end_state: NettingChannelEndState, amount: TokenAmount) -> bool: ...

def is_valid_signature(
    data: bytes,
    signature: Signature,
    sender_address: Address,
) -> SuccessOrError: ...

def is_valid_balanceproof_signature(
    balance_proof: BalanceProofSignedState,
    sender_address: Address,
) -> SuccessOrError: ...

def is_balance_proof_usable_onchain(
    received_balance_proof: BalanceProofSignedState,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
) -> SuccessOrError: ...

def is_valid_lockedtransfer(
    transfer_state: LockedTransferSignedState,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
) -> PendingLocksStateOrError: ...

def is_valid_lock_expired(
    state_change: ReceiveLockExpired,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
    block_number: BlockNumber,
) -> PendingLocksStateOrError: ...

def valid_lockedtransfer_check(
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
    message_name: str,
    received_balance_proof: BalanceProofSignedState,
    lock: HashTimeLockState,
) -> PendingLocksStateOrError: ...

def refund_transfer_matches_transfer(
    refund_transfer: LockedTransferSignedState,
    transfer: LockedTransferUnsignedState,
) -> bool: ...

def is_valid_refund(
    refund: ReceiveTransferRefund,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
    receiver_state: NettingChannelEndState,
    received_transfer: LockedTransferUnsignedState,
) -> PendingLocksStateOrError: ...

def is_valid_unlock(
    unlock: ReceiveUnlock,
    channel_state: NettingChannelState,
    sender_state: NettingChannelEndState,
) -> PendingLocksStateOrError: ...

def is_valid_total_withdraw(
    channel_state: NettingChannelState,
    our_total_withdraw: WithdrawAmount,
    allow_zero: bool = ...,
) -> SuccessOrError: ...

def is_valid_action_coopsettle(
    channel_state: NettingChannelState,
    coop_settle: ActionChannelCoopSettle,
    total_withdraw: WithdrawAmount,
) -> SuccessOrError: ...

def is_valid_withdraw_request(
    channel_state: NettingChannelState,
    withdraw_request: ReceiveWithdrawRequest,
) -> SuccessOrError: ...

def is_valid_withdraw_confirmation(
    channel_state: NettingChannelState,
    received_withdraw: ReceiveWithdrawConfirmation,
) -> SuccessOrError: ...

def is_valid_withdraw_expired(
    channel_state: NettingChannelState,
    state_change: ReceiveWithdrawExpired,
    withdraw_state: PendingWithdrawState,
    block_number: BlockNumber,
) -> SuccessOrError: ...

def get_amount_unclaimed_onchain(end_state: NettingChannelEndState) -> TokenAmount: ...

def get_amount_locked(end_state: NettingChannelEndState) -> LockedAmount: ...

def get_capacity(channel_state: NettingChannelState) -> TokenAmount: ...

def get_balance(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> Balance: ...

def _get_balance(
    sender: NettingChannelEndState,
    receiver: NettingChannelEndState,
    subtract_withdraws: bool = ...,
) -> Balance: ...

def get_max_withdraw_amount(
    sender: NettingChannelEndState,
    receiver: NettingChannelEndState,
) -> WithdrawAmount: ...

def get_current_balanceproof(end_state: NettingChannelEndState) -> BalanceProofData: ...

def get_current_nonce(end_state: NettingChannelEndState) -> Nonce: ...

def get_distributable(
    sender: NettingChannelEndState,
    receiver: NettingChannelEndState,
) -> TokenAmount: ...

def get_lock(
    end_state: NettingChannelEndState,
    secrethash: SecretHash,
) -> Optional[HashTimeLockState]: ...

def lock_exists_in_either_channel_side(
    channel_state: NettingChannelState,
    secrethash: SecretHash,
) -> bool: ...

def get_next_nonce(end_state: NettingChannelEndState) -> Nonce: ...

def get_number_of_pending_transfers(channel_end_state: NettingChannelEndState) -> int: ...

def get_status(channel_state: NettingChannelState) -> str: ...

def _del_unclaimed_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None: ...

def _del_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None: ...

def set_closed(channel_state: NettingChannelState, block_number: BlockNumber) -> None: ...

def set_settled(channel_state: NettingChannelState, block_number: BlockNumber) -> None: ...

def set_coop_settled(channel_end_state: NettingChannelEndState, block_number: BlockNumber) -> None: ...

def update_contract_balance(end_state: NettingChannelEndState, contract_balance: Balance) -> None: ...

def compute_locks_with(
    locks: PendingLocksState,
    lock: HashTimeLockState,
) -> Optional[PendingLocksState]: ...

def compute_locks_without(
    locks: PendingLocksState,
    lock_encoded: EncodedData,
) -> Optional[PendingLocksState]: ...

def compute_locksroot(locks: PendingLocksState) -> Locksroot: ...

def create_sendlockedtransfer(
    channel_state: NettingChannelState,
    initiator: InitiatorAddress,
    target: TargetAddress,
    amount: PaymentWithFeeAmount,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    expiration: BlockExpiration,
    secret: Optional[Secret],
    secrethash: SecretHash,
    route_states: List[RouteState],
    recipient_metadata: Optional[AddressMetadata] = ...,
    previous_metadata: Optional[Any] = ...,
) -> Tuple[SendLockedTransfer, PendingLocksState]: ...

def create_unlock(
    channel_state: NettingChannelState,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    secret: Secret,
    lock: HashTimeLockState,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = ...,
) -> SendUnlockAndPendingLocksState: ...

def send_lockedtransfer(
    channel_state: NettingChannelState,
    initiator: InitiatorAddress,
    target: TargetAddress,
    amount: PaymentWithFeeAmount,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    expiration: BlockExpiration,
    secret: Optional[Secret],
    secrethash: SecretHash,
    route_states: List[RouteState],
    recipient_metadata: Optional[AddressMetadata] = ...,
    previous_metadata: Optional[Any] = ...,
) -> SendLockedTransfer: ...

def send_unlock(
    channel_state: NettingChannelState,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    secret: Secret,
    secrethash: SecretHash,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = ...,
) -> SendUnlock: ...

def events_for_close(
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[Event]: ...

def send_withdraw_request(
    channel_state: NettingChannelState,
    total_withdraw: WithdrawAmount,
    expiration: BlockExpiration,
    pseudo_random_generator: random.Random,
    recipient_metadata: Optional[AddressMetadata],
    coop_settle: bool = ...,
) -> List[Event]: ...

def create_sendexpiredlock(
    sender_end_state: NettingChannelEndState,
    locked_lock: HashTimeLockState,
    pseudo_random_generator: random.Random,
    chain_id: ChainID,
    token_network_address: TokenNetworkAddress,
    channel_identifier: ChannelID,
    recipient: Address,
    recipient_metadata: Optional[AddressMetadata] = ...,
) -> Tuple[Optional[SendLockExpired], Optional[PendingLocksState]]: ...

def send_lock_expired(
    channel_state: NettingChannelState,
    locked_lock: HashTimeLockState,
    pseudo_random_generator: random.Random,
    recipient_metadata: Optional[AddressMetadata] = ...,
) -> List[SendLockExpired]: ...

def events_for_expired_withdraws(
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    pseudo_random_generator: random.Random,
) -> List[Event]: ...

def register_secret_endstate(
    end_state: NettingChannelEndState,
    secret: Secret,
    secrethash: SecretHash,
) -> None: ...

def register_onchain_secret_endstate(
    end_state: NettingChannelEndState,
    secret: Secret,
    secrethash: SecretHash,
    secret_reveal_block_number: BlockNumber,
    delete_lock: bool = ...,
) -> None: ...

def register_offchain_secret(
    channel_state: NettingChannelState,
    secret: Secret,
    secrethash: SecretHash,
) -> None: ...

def register_onchain_secret(
    channel_state: NettingChannelState,
    secret: Secret,
    secrethash: SecretHash,
    secret_reveal_block_number: BlockNumber,
    delete_lock: bool = ...,
) -> None: ...

def handle_state_transitions(
    action: StateChange,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult[Optional[NettingChannelState]]: ...

def handle_refundtransfer(
    received_transfer: LockedTransferUnsignedState,
    channel_state: NettingChannelState,
    refund: ReceiveTransferRefund,
) -> Tuple[bool, List[Event], Optional[str]]: ...

def handle_receive_lock_expired(
    channel_state: NettingChannelState,
    state_change: ReceiveLockExpired,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = ...,
) -> TransitionResult[NettingChannelState]: ...

def handle_receive_lockedtransfer(
    channel_state: NettingChannelState,
    mediated_transfer: LockedTransferSignedState,
    recipient_metadata: Optional[AddressMetadata] = ...,
) -> Tuple[bool, List[Event], Optional[str]]: ...

def handle_unlock(
    channel_state: NettingChannelState,
    unlock: ReceiveUnlock,
    recipient_metadata: Optional[AddressMetadata] = ...,
) -> Tuple[bool, List[Event], Optional[str]]: ...

def events_for_coop_settle(
    channel_state: NettingChannelState,
    coop_settle_state: CoopSettleState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[Event]: ...

def update_fee_schedule_after_balance_change(
    channel_state: NettingChannelState,
    fee_config: MediationFeeConfig,
) -> List[Event]: ...

def sanity_check(channel_state: NettingChannelState) -> None: ...

def state_transition(
    channel_state: NettingChannelState,
    state_change: StateChange,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult[Optional[NettingChannelState]]: ...