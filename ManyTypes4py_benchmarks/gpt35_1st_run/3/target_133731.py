from raiden.transfer.state import NettingChannelState
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, ReceiveUnlock, ReceiveLockExpired
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveSecretReveal
from raiden.transfer.utils import is_valid_secret_reveal
from raiden.utils.typing import MYPY_ANNOTATION, Address, BlockHash, BlockNumber, List, Optional, PaymentAmount

def sanity_check(old_state: Optional[TargetTransferState], new_state: Optional[TargetTransferState], channel_state: NettingChannelState) -> None:
    ...

def events_for_onchain_secretreveal(target_state: TargetTransferState, channel_state: NettingChannelState, block_number: BlockNumber, block_hash: BlockHash) -> List[Event]:
    ...

def handle_inittarget(state_change: ActionInitTarget, channel_state: NettingChannelState, pseudo_random_generator, block_number: BlockNumber) -> TransitionResult:
    ...

def handle_offchain_secretreveal(target_state: TargetTransferState, state_change: ReceiveSecretReveal, channel_state: NettingChannelState, pseudo_random_generator, block_number: BlockNumber) -> TransitionResult:
    ...

def handle_onchain_secretreveal(target_state: TargetTransferState, state_change: ContractReceiveSecretReveal, channel_state: NettingChannelState) -> TransitionResult:
    ...

def handle_unlock(target_state: TargetTransferState, state_change: ReceiveUnlock, channel_state: NettingChannelState) -> TransitionResult:
    ...

def handle_block(target_state: TargetTransferState, channel_state: NettingChannelState, block_number: BlockNumber, block_hash: BlockHash) -> TransitionResult:
    ...

def handle_lock_expired(target_state: TargetTransferState, state_change: ReceiveLockExpired, channel_state: NettingChannelState, block_number: BlockNumber) -> TransitionResult:
    ...

def state_transition(target_state: Optional[TargetTransferState], state_change, channel_state: NettingChannelState, pseudo_random_generator, block_number: BlockNumber) -> TransitionResult:
    ...
