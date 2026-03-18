```pyi
from typing import Any, Dict, List, Optional, Tuple
from raiden.transfer.architecture import TransitionResult
from raiden.transfer.events import EventInvalidSecretRequest, EventPaymentSentFailed, EventPaymentSentSuccess
from raiden.transfer.mediated_transfer.events import EventRouteFailed, EventUnlockFailed, EventUnlockSuccess, SendLockedTransfer, SendSecretReveal
from raiden.transfer.mediated_transfer.state import InitiatorTransferState, TransferDescriptionWithSecretState
from raiden.transfer.state import ChannelState, NettingChannelState, RouteState
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, StateChange
from raiden.utils.typing import Address, BlockExpiration, BlockNumber, FeeAmount, MessageID, PaymentAmount, PaymentWithFeeAmount, Secret, SecretHash, TokenNetworkAddress

def calculate_fee_margin(payment_amount: PaymentAmount, estimated_fee: FeeAmount) -> FeeAmount: ...
def calculate_safe_amount_with_fee(payment_amount: PaymentAmount, estimated_fee: FeeAmount) -> PaymentWithFeeAmount: ...
def events_for_unlock_lock(
    initiator_state: InitiatorTransferState,
    channel_state: NettingChannelState,
    secret: Secret,
    secrethash: SecretHash,
    pseudo_random_generator: Any,
    block_number: BlockNumber,
) -> List[Any]: ...
def handle_block(
    initiator_state: InitiatorTransferState,
    state_change: Block,
    channel_state: NettingChannelState,
    pseudo_random_generator: Any,
) -> TransitionResult: ...
def try_new_route(
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], NettingChannelState],
    candidate_route_states: List[RouteState],
    transfer_description: TransferDescriptionWithSecretState,
    pseudo_random_generator: Any,
    block_number: BlockNumber,
) -> TransitionResult: ...
def send_lockedtransfer(
    transfer_description: TransferDescriptionWithSecretState,
    channel_state: NettingChannelState,
    message_identifier: MessageID,
    block_number: BlockNumber,
    route_state: RouteState,
    route_states: List[RouteState],
) -> SendLockedTransfer: ...
def handle_secretrequest(
    initiator_state: InitiatorTransferState,
    state_change: Any,
    channel_state: NettingChannelState,
    pseudo_random_generator: Any,
) -> TransitionResult: ...
def handle_offchain_secretreveal(
    initiator_state: InitiatorTransferState,
    state_change: Any,
    channel_state: NettingChannelState,
    pseudo_random_generator: Any,
    block_number: BlockNumber,
) -> TransitionResult: ...
def handle_onchain_secretreveal(
    initiator_state: InitiatorTransferState,
    state_change: ContractReceiveSecretReveal,
    channel_state: NettingChannelState,
    pseudo_random_generator: Any,
    block_number: BlockNumber,
) -> TransitionResult: ...
def state_transition(
    initiator_state: InitiatorTransferState,
    state_change: StateChange,
    channel_state: NettingChannelState,
    pseudo_random_generator: Any,
    block_number: BlockNumber,
) -> TransitionResult: ...
```