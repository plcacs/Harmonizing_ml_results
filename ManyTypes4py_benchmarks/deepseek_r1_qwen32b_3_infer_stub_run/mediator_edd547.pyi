import itertools
import operator
import random
from fractions import Fraction
from typing import Any, Callable, List, Optional, Tuple, Union
from raiden.exceptions import UndefinedMediationFee
from raiden.transfer import channel, routes, secret_registry
from raiden.transfer.architecture import Event, StateChange, SuccessOrError, TransitionResult
from raiden.transfer.mediated_transfer.events import (
    EventUnexpectedSecretReveal,
    EventUnlockClaimFailed,
    EventUnlockClaimSuccess,
    EventUnlockFailed,
    EventUnlockSuccess,
    SendLockedTransfer,
    SendSecretReveal,
)
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
from raiden.transfer.state import ChannelState, NettingChannelState
from raiden.utils.typing import (
    Address,
    BlockExpiration,
    BlockHash,
    BlockNumber,
    BlockTimeout,
    ChannelID,
    Dict,
    List,
    LockType,
    Optional,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    TokenAmount,
    TokenNetworkAddress,
    Tuple,
    Union,
)

STATE_SECRET_KNOWN = ('payee_secret_revealed', 'payee_contract_unlock', 'payee_balance_proof', 'payer_secret_revealed', 'payer_waiting_unlock', 'payer_balance_proof')
STATE_TRANSFER_PAID = ('payee_contract_unlock', 'payee_balance_proof', 'payer_balance_proof')
STATE_TRANSFER_FINAL = ('payee_contract_unlock', 'payee_balance_proof', 'payee_expired', 'payer_balance_proof', 'payer_expired')

def is_lock_valid(expiration: BlockExpiration, block_number: BlockNumber) -> bool: ...
def is_safe_to_wait(lock_expiration: BlockExpiration, reveal_timeout: BlockTimeout, block_number: BlockNumber) -> Union[SuccessOrError, None]: ...
def is_send_transfer_almost_equal(send_channel: ChannelState, send: LockedTransferSignedState, received: LockedTransferUnsignedState) -> bool: ...
def has_secret_registration_started(channel_states: List[ChannelState], transfers_pair: List[MediationPairState], secrethash: SecretHash) -> bool: ...
def get_payee_channel(channelidentifiers_to_channels: Dict[ChannelID, ChannelState], transfer_pair: MediationPairState) -> Optional[ChannelState]: ...
def get_payer_channel(channelidentifiers_to_channels: Dict[ChannelID, ChannelState], transfer_pair: MediationPairState) -> Optional[ChannelState]: ...
def get_pending_transfer_pairs(transfers_pair: List[MediationPairState]) -> List[MediationPairState]: ...
def find_intersection(fee_func: FeeScheduleState, line: Callable[[int], float]) -> Optional[float]: ...
def get_amount_without_fees(amount_with_fees: PaymentWithFeeAmount, channel_in: ChannelState, channel_out: ChannelState) -> Optional[PaymentWithFeeAmount]: ...
def sanity_check(state: MediatorTransferState, channelidentifiers_to_channels: Dict[ChannelID, ChannelState]) -> None: ...
def clear_if_finalized(iteration: TransitionResult, channelidentifiers_to_channels: Dict[ChannelID, ChannelState]) -> TransitionResult: ...

def forward_transfer_pair(
    payer_transfer: LockedTransferUnsignedState,
    payer_channel: ChannelState,
    payee_channel: ChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> Tuple[Optional[MediationPairState], List[SendLockedTransfer]]: ...

def set_offchain_secret(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    secret: Secret,
    secrethash: SecretHash,
) -> List[Event]: ...

def set_onchain_secret(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    secret: Secret,
    secrethash: SecretHash,
    block_number: BlockNumber,
) -> List[Event]: ...

def set_offchain_reveal_state(transfers_pair: List[MediationPairState], payee_address: Address) -> None: ...
def events_for_expired_pairs(
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    transfers_pair: List[MediationPairState],
    waiting_transfer: Optional[WaitingTransferState],
    block_number: BlockNumber,
) -> List[Event]: ...

def events_for_secretreveal(
    transfers_pair: List[MediationPairState],
    secret: Secret,
    pseudo_random_generator: random.Random,
) -> List[SendSecretReveal]: ...

def events_for_balanceproof(
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    transfers_pair: List[MediationPairState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    secret: Secret,
    secrethash: SecretHash,
) -> List[Event]: ...

def events_for_onchain_secretreveal_if_dangerzone(
    channelmap: Dict[ChannelID, ChannelState],
    secrethash: SecretHash,
    transfers_pair: List[MediationPairState],
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[Event]: ...

def events_for_onchain_secretreveal_if_closed(
    channelmap: Dict[ChannelID, ChannelState],
    transfers_pair: List[MediationPairState],
    secret: Secret,
    secrethash: SecretHash,
    block_hash: BlockHash,
) -> List[Event]: ...

def events_to_remove_expired_locks(
    mediator_state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    block_number: BlockNumber,
    pseudo_random_generator: random.Random,
) -> List[Event]: ...

def secret_learned(
    state: MediatorTransferState,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    block_hash: BlockHash,
    secret: Secret,
    secrethash: SecretHash,
    payee_address: Address,
) -> TransitionResult: ...

def mediate_transfer(
    state: MediatorTransferState,
    payer_channel: ChannelState,
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], ChannelState],
    pseudo_random_generator: random.Random,
    payer_transfer: LockedTransferUnsignedState,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_init(
    state_change: ActionInitMediator,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], ChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_block(
    mediator_state: MediatorTransferState,
    state_change: Block,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], ChannelState],
    pseudo_random_generator: random.Random,
) -> TransitionResult: ...

def handle_refundtransfer(
    mediator_state: MediatorTransferState,
    mediator_state_change: ReceiveTransferRefund,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], ChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_offchain_secretreveal(
    mediator_state: MediatorTransferState,
    mediator_state_change: ReceiveSecretReveal,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> TransitionResult: ...

def handle_onchain_secretreveal(
    mediator_state: MediatorTransferState,
    onchain_secret_reveal: ContractReceiveSecretReveal,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_unlock(
    mediator_state: MediatorTransferState,
    state_change: ReceiveUnlock,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
) -> TransitionResult: ...

def handle_lock_expired(
    mediator_state: MediatorTransferState,
    state_change: ReceiveLockExpired,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    block_number: BlockNumber,
) -> TransitionResult: ...

def state_transition(
    mediator_state: Optional[MediatorTransferState],
    state_change: StateChange,
    channelidentifiers_to_channels: Dict[ChannelID, ChannelState],
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], ChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> TransitionResult: ...