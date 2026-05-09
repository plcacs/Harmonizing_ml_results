from typing import Dict, List, Optional, Tuple, Union
from raiden.transfer import channel, routes
from raiden.transfer.architecture import Event, StateChange, TransitionResult
from raiden.transfer.events import (
    EventPaymentSentFailed,
    EventRouteFailed,
    EventUnlockClaimFailed,
    EventUnlockFailed,
)
from raiden.transfer.mediated_transfer.state import (
    InitiatorPaymentState,
    InitiatorTransferState,
    TransferDescriptionWithSecretState,
)
from raiden.transfer.mediated_transfer.state_change import (
    ActionInitInitiator,
    ActionTransferReroute,
    ReceiveLockExpired,
    ReceiveSecretRequest,
    ReceiveSecretReveal,
    ReceiveTransferCancelRoute,
)
from raiden.transfer.state import NettingChannelState, RouteState
from raiden.transfer.state_change import ActionCancelPayment, Block, ContractReceiveSecretReveal
from raiden.utils.typing import (
    Address,
    BlockNumber,
    ChannelID,
    MYPY_ANNOTATION,
    Optional,
    SecretHash,
    TokenNetworkAddress,
)

def clear_if_finalized(iteration: TransitionResult) -> TransitionResult: ...
def transfer_exists(payment_state: InitiatorPaymentState, secrethash: SecretHash) -> bool: ...
def cancel_other_transfers(payment_state: InitiatorPaymentState) -> None: ...
def can_cancel(initiator: InitiatorTransferState) -> bool: ...
def events_for_cancel_current_route(
    route_state: RouteState,
    transfer_description: TransferDescriptionWithSecretState,
) -> List[Union[EventUnlockFailed, EventRouteFailed]]: ...
def cancel_current_route(
    payment_state: InitiatorPaymentState,
    initiator_state: InitiatorTransferState,
) -> List[Union[EventUnlockFailed, EventRouteFailed]]: ...

def subdispatch_to_initiatortransfer(
    payment_state: InitiatorPaymentState,
    initiator_state: InitiatorTransferState,
    state_change: StateChange,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def subdispatch_to_all_initiatortransfer(
    payment_state: InitiatorPaymentState,
    state_change: StateChange,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_block(
    payment_state: InitiatorPaymentState,
    state_change: Block,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_init(
    payment_state: Optional[InitiatorPaymentState],
    state_change: ActionInitInitiator,
    addresses_to_channel: Dict[Address, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_cancelpayment(
    payment_state: InitiatorPaymentState,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
) -> TransitionResult: ...

def handle_failroute(
    payment_state: InitiatorPaymentState,
    state_change: ReceiveTransferCancelRoute,
) -> TransitionResult: ...

def handle_transferreroute(
    payment_state: InitiatorPaymentState,
    state_change: ActionTransferReroute,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    addresses_to_channel: Dict[Address, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_lock_expired(
    payment_state: InitiatorPaymentState,
    state_change: ReceiveLockExpired,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_offchain_secretreveal(
    payment_state: InitiatorPaymentState,
    state_change: ReceiveSecretReveal,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_onchain_secretreveal(
    payment_state: InitiatorPaymentState,
    state_change: ContractReceiveSecretReveal,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def handle_secretrequest(
    payment_state: InitiatorPaymentState,
    state_change: ReceiveSecretRequest,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...

def state_transition(
    payment_state: Optional[InitiatorPaymentState],
    state_change: StateChange,
    channelidentifiers_to_channels: Dict[ChannelID, NettingChannelState],
    addresses_to_channel: Dict[Address, NettingChannelState],
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult: ...