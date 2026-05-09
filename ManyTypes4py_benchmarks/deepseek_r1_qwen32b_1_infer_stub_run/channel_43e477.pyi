import random
from enum import Enum
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast
from eth_utils import Address, BlockHash, ChainID, EncodedData, HexStr, Nonce, Secret, SecretHash, Signature
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
from raiden.transfer.mediated_transfer.events import SendLockedTransfer, SendLockExpired, SendUnlock
from raiden.transfer.mediated_transfer.state import (
    LockedTransferSignedState,
    LockedTransferUnsignedState,
)
from raiden.transfer.mediated_transfer.state_change import ReceiveLockExpired, ReceiveTransferRefund
from raiden.transfer.state import (
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
    Balance,
    BlockExpiration,
    BlockNumber,
    ChainID,
    ChannelID,
    EncodedData,
    FeeAmount,
    InitiatorAddress,
    Locksroot,
    LockedAmount,
    MessageID,
    PaymentAmount,
    PaymentID,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    TargetAddress,
    TokenAmount,
    TokenNetworkAddress,
    WithdrawAmount,
)

UnlockGain = NamedTuple('UnlockGain', [])

class ChannelUsability(Enum):
    USABLE = True
    NOT_OPENED = str
    INVALID_SETTLE_TIMEOUT = str
    CHANNEL_REACHED_PENDING_LIMIT = str
    CHANNEL_DOESNT_HAVE_ENOUGH_DISTRIBUTABLE = str
    CHANNEL_BALANCE_PROOF_WOULD_OVERFLOW = str
    LOCKTIMEOUT_MISMATCH = str

def get_safe_initial_expiration(block_number: BlockNumber, reveal_timeout: BlockTimeout, lock_timeout: Optional[BlockTimeout]) -> BlockExpiration:
    ...

def get_sender_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    ...

def get_receiver_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    ...

def is_channel_usable_for_mediation(channel_state: NettingChannelState, transfer_amount: TokenAmount, lock_timeout: BlockTimeout) -> bool:
    ...

def is_channel_usable_for_new_transfer(channel_state: NettingChannelState, transfer_amount: TokenAmount, lock_timeout: BlockTimeout) -> ChannelUsability:
    ...

def is_lock_pending(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    ...

def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    ...

def is_lock_expired(end_state: NettingChannelEndState, lock: HashTimeLockState, block_number: BlockNumber, lock_expiration_threshold: BlockExpiration) -> SuccessOrError:
    ...

def is_transfer_expired(transfer: LockedTransferSignedState, affected_channel: NettingChannelState, block_number: BlockNumber) -> bool:
    ...

def is_withdraw_expired(block_number: BlockNumber, expiration_threshold: BlockExpiration) -> bool:
    ...

def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    ...

def is_secret_known_offchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    ...

def is_secret_known_onchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    ...

def is_valid_channel_total_withdraw(channel_total_withdraw: TokenAmount) -> bool:
    ...

def is_valid_withdraw(withdraw_request: Any) -> bool:
    ...

def get_secret(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[Secret]:
    ...

def is_balance_proof_safe_for_onchain_operations(balance_proof: Any) -> bool:
    ...

def is_valid_amount(end_state: NettingChannelEndState, amount: TokenAmount) -> bool:
    ...

def is_valid_signature(data: EncodedData, signature: Signature, sender_address: Address) -> SuccessOrError:
    ...

def is_valid_balanceproof_signature(balance_proof: Any, sender_address: Address) -> SuccessOrError:
    ...

def is_balance_proof_usable_onchain(received_balance_proof: Any, channel_state: NettingChannelState, sender_state: NettingChannelEndState) -> SuccessOrError:
    ...

def is_valid_lockedtransfer(transfer_state: Any, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState) -> bool:
    ...

def is_valid_lock_expired(state_change: Any, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, block_number: BlockNumber) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    ...

def valid_lockedtransfer_check(channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, message_name: str, received_balance_proof: Any, lock: HashTimeLockState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    ...

def refund_transfer_matches_transfer(refund_transfer: Any, transfer: Any) -> bool:
    ...

def is_valid_refund(refund: Any, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, received_transfer: Any) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    ...

def is_valid_unlock(unlock: Any, channel_state: NettingChannelState, sender_state: NettingChannelEndState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    ...

def is_valid_total_withdraw(channel_state: NettingChannelState, our_total_withdraw: TokenAmount, allow_zero: bool = False) -> SuccessOrError:
    ...

def is_valid_action_coopsettle(channel_state: NettingChannelState, coop_settle: Any, total_withdraw: TokenAmount) -> SuccessOrError:
    ...

def is_valid_withdraw_request(channel_state: NettingChannelState, withdraw_request: Any) -> SuccessOrError:
    ...

def is_valid_withdraw_confirmation(channel_state: NettingChannelState, received_withdraw: Any) -> SuccessOrError:
    ...

def is_valid_withdraw_expired(channel_state: NettingChannelState, state_change: Any, withdraw_state: ExpiredWithdrawState, block_number: BlockNumber) -> SuccessOrError:
    ...

def get_amount_unclaimed_onchain(end_state: NettingChannelEndState) -> TokenAmount:
    ...

def get_amount_locked(end_state: NettingChannelEndState) -> LockedAmount:
    ...

def get_capacity(channel_state: NettingChannelState) -> TokenAmount:
    ...

def get_balance(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> Balance:
    ...

def get_max_withdraw_amount(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> WithdrawAmount:
    ...

def get_current_balanceproof(end_state: NettingChannelEndState) -> Tuple[Locksroot, Nonce, TokenAmount, LockedAmount]:
    ...

def get_current_nonce(end_state: NettingChannelEndState) -> Nonce:
    ...

def get_distributable(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> TokenAmount:
    ...

def get_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[HashTimeLockState]:
    ...

def lock_exists_in_either_channel_side(channel_state: NettingChannelState, secrethash: SecretHash) -> bool:
    ...

def get_next_nonce(end_state: NettingChannelEndState) -> Nonce:
    ...

def get_number_of_pending_transfers(channel_end_state: NettingChannelEndState) -> int:
    ...

def get_status(channel_state: NettingChannelState) -> ChannelState:
    ...

def _del_unclaimed_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None:
    ...

def _del_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None:
    ...

def set_closed(channel_state: NettingChannelState, block_number: BlockNumber) -> None:
    ...

def set_settled(channel_state: NettingChannelState, block_number: BlockNumber) -> None:
    ...

def set_coop_settled(channel_end_state: NettingChannelEndState, block_number: BlockNumber) -> None:
    ...

def update_contract_balance(end_state: NettingChannelEndState, contract_balance: Balance) -> None:
    ...

def compute_locks_with(locks: PendingLocksState, lock: HashTimeLockState) -> Optional[PendingLocksState]:
    ...

def compute_locks_without(locks: PendingLocksState, lock_encoded: EncodedData) -> Optional[PendingLocksState]:
    ...

def compute_locksroot(locks: PendingLocksState) -> Locksroot:
    ...

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
    ...

def create_unlock(
    channel_state: NettingChannelState,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    secret: Secret,
    lock: HashTimeLockState,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> Tuple[SendUnlock, PendingLocksState]:
    ...

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
    ...

def send_unlock(
    channel_state: NettingChannelState,
    message_identifier: MessageID,
    payment_identifier: PaymentID,
    secret: Secret,
    secrethash: SecretHash,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> SendUnlock:
    ...

def events_for_close(channel_state: NettingChannelState, block_number: BlockNumber, block_hash: BlockHash) -> List[ContractSendChannelClose]:
    ...

def send_withdraw_request(
    channel_state: NettingChannelState,
    total_withdraw: TokenAmount,
    expiration: BlockExpiration,
    pseudo_random_generator: random.Random,
    recipient_metadata: Optional[AddressMetadata],
    coop_settle: bool = False,
) -> List[SendWithdrawRequest]:
    ...

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
    ...

def send_lock_expired(
    channel_state: NettingChannelState,
    locked_lock: HashTimeLockState,
    pseudo_random_generator: random.Random,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> List[SendLockExpired]:
    ...

def events_for_expired_withdraws(
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    pseudo_random_generator: random.Random,
) -> List[SendWithdrawExpired]:
    ...

def register_secret_endstate(end_state: NettingChannelEndState, secret: Secret, secrethash: SecretHash) -> None:
    ...

def register_onchain_secret_endstate(
    end_state: NettingChannelEndState,
    secret: Secret,
    secrethash: SecretHash,
    secret_reveal_block_number: BlockNumber,
    delete_lock: bool = True,
) -> None:
    ...

def register_offchain_secret(channel_state: NettingChannelState, secret: Secret, secrethash: SecretHash) -> None:
    ...

def register_onchain_secret(
    channel_state: NettingChannelState,
    secret: Secret,
    secrethash: SecretHash,
    secret_reveal_block_number: BlockNumber,
    delete_lock: bool = True,
) -> None:
    ...

@singledispatch
def handle_state_transitions(
    action: StateChange,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_action_close(
    action: ActionChannelClose,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_action_coop_settle(
    action: ActionChannelCoopSettle,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_action_withdraw(
    action: ActionChannelWithdraw,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_action_set_reveal_timeout(
    action: ActionChannelSetRevealTimeout,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    ...

def events_for_coop_settle(
    channel_state: NettingChannelState,
    coop_settle_state: CoopSettleState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[ContractSendChannelCoopSettle]:
    ...

@handle_state_transitions.register
def _handle_receive_withdraw_request(
    action: ReceiveWithdrawRequest,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_receive_withdraw_confirmation(
    action: ReceiveWithdrawConfirmation,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_receive_withdraw_expired(
    action: ReceiveWithdrawExpired,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    ...

def handle_refundtransfer(
    received_transfer: Any,
    channel_state: NettingChannelState,
    refund: Any,
) -> Tuple[bool, List[Event], Optional[str]]:
    ...

def handle_receive_lock_expired(
    channel_state: NettingChannelState,
    state_change: Any,
    block_number: BlockNumber,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> TransitionResult:
    ...

def handle_receive_lockedtransfer(
    channel_state: NettingChannelState,
    mediated_transfer: Any,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> Tuple[bool, List[Event], Optional[str]]:
    ...

def handle_unlock(
    channel_state: NettingChannelState,
    unlock: Any,
    recipient_metadata: Optional[AddressMetadata] = None,
) -> Tuple[bool, List[Event], Optional[str]]:
    ...

@handle_state_transitions.register
def _handle_block(
    action: Block,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    pseudo_random_generator: random.Random,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_channel_closed(
    action: ContractReceiveChannelClosed,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_channel_updated_transfer(
    action: ContractReceiveChannelUpdateTransfer,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_channel_settled(
    action: ContractReceiveChannelSettled,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    ...

def update_fee_schedule_after_balance_change(
    channel_state: NettingChannelState,
    fee_config: MediationFeeConfig,
) -> List[Event]:
    ...

@handle_state_transitions.register
def _handle_channel_deposit(
    action: ContractReceiveChannelDeposit,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_channel_withdraw(
    action: ContractReceiveChannelWithdraw,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    ...

@handle_state_transitions.register
def _handle_channel_batch_unlock(
    action: ContractReceiveChannelBatchUnlock,
    channel_state: NettingChannelState,
    **kwargs: Any,
) -> TransitionResult:
    ...

def sanity_check(channel_state: NettingChannelState) -> None:
    ...

def state_transition(
    channel_state: NettingChannelState,
    state_change: StateChange,
    block_number: BlockNumber,
    block_hash: BlockHash,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    ...