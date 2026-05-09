from raiden.transfer import channel, secret_registry
from raiden.transfer.architecture import Event, StateChange, TransitionResult
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import (
    ActionInitTarget,
    ReceiveLockExpired,
    ReceiveSecretReveal,
    ContractReceiveSecretReveal,
    ReceiveUnlock,
)
from raiden.transfer.state import NettingChannelState
from raiden.utils.typing import (
    Address,
    BlockHash,
    BlockNumber,
    List,
    MYPY_ANNOTATION,
    Optional,
    PaymentAmount,
    Random,
)

def sanity_check(old_state: Optional[TargetTransferState], new_state: Optional[TargetTransferState], channel_state: NettingChannelState) -> None: ...

def events_for_onchain_secretreveal(target_state: TargetTransferState, channel_state: NettingChannelState, block_number: BlockNumber, block_hash: BlockHash) -> List[Event]: ...

def handle_inittarget(
    state_change: ActionInitTarget,
    channel_state: NettingChannelState,
    pseudo_random_generator: Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState]]: ...

def handle_offchain_secretreveal(
    target_state: TargetTransferState,
    state_change: ReceiveSecretReveal,
    channel_state: NettingChannelState,
    pseudo_random_generator: Random,
    block_number: BlockNumber,
) -> TransitionResult[TargetTransferState]: ...

def handle_onchain_secretreveal(
    target_state: TargetTransferState,
    state_change: ContractReceiveSecretReveal,
    channel_state: NettingChannelState,
) -> TransitionResult[TargetTransferState]: ...

def handle_unlock(
    target_state: TargetTransferState,
    state_change: ReceiveUnlock,
    channel_state: NettingChannelState,
) -> TransitionResult[Optional[TargetTransferState]]: ...

def handle_block(
    target_state: TargetTransferState,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> TransitionResult[TargetTransferState]: ...

def handle_lock_expired(
    target_state: TargetTransferState,
    state_change: ReceiveLockExpired,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState]]: ...

def state_transition(
    target_state: Optional[TargetTransferState],
    state_change: StateChange,
    channel_state: NettingChannelState,
    pseudo_random_generator: Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState]]: ...