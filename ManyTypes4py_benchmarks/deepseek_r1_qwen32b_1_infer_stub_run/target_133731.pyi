import random
from raiden.transfer import channel, secret_registry
from raiden.transfer.architecture import Event, StateChange, TransitionResult
from raiden.transfer.events import EventPaymentReceivedSuccess
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE
from raiden.transfer.mediated_transfer.events import (
    EventUnlockClaimFailed,
    EventUnlockClaimSuccess,
    SendSecretRequest,
    SendSecretReveal,
)
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.state import NettingChannelState
from raiden.utils.typing import (
    MYPY_ANNOTATION,
    Address,
    BlockHash,
    BlockNumber,
    List,
    Optional,
    PaymentAmount,
)

def sanity_check(
    old_state: Optional[TargetTransferState],
    new_state: Optional[TargetTransferState],
    channel_state: NettingChannelState,
) -> None:
    ...

def events_for_onchain_secretreveal(
    target_state: TargetTransferState,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[secret_registry.SecretRevealOnChainEvent]:
    ...

def handle_inittarget(
    state_change: ActionInitTarget,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState]]:
    ...

def handle_offchain_secretreveal(
    target_state: TargetTransferState,
    state_change: ReceiveSecretReveal,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[TargetTransferState]:
    ...

def handle_onchain_secretreveal(
    target_state: TargetTransferState,
    state_change: ContractReceiveSecretReveal,
    channel_state: NettingChannelState,
) -> TransitionResult[TargetTransferState]:
    ...

def handle_unlock(
    target_state: TargetTransferState,
    state_change: ReceiveUnlock,
    channel_state: NettingChannelState,
) -> TransitionResult[Optional[TargetTransferState]]:
    ...

def handle_block(
    target_state: TargetTransferState,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> TransitionResult[TargetTransferState]:
    ...

def handle_lock_expired(
    target_state: TargetTransferState,
    state_change: ReceiveLockExpired,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState]]:
    ...

def state_transition(
    target_state: Optional[TargetTransferState],
    state_change: StateChange,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState]]:
    ...