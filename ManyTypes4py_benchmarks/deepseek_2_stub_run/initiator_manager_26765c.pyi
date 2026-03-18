```python
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from raiden.transfer.architecture import Event, StateChange, TransitionResult
from raiden.transfer.events import EventPaymentSentFailed
from raiden.transfer.mediated_transfer.events import (
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
    SecretHash,
    TokenNetworkAddress,
)

def clear_if_finalized(iteration: Any) -> Any: ...
def transfer_exists(payment_state: Any, secrethash: Any) -> Any: ...
def cancel_other_transfers(payment_state: Any) -> None: ...
def can_cancel(initiator: Any) -> Any: ...
def events_for_cancel_current_route(route_state: Any, transfer_description: Any) -> List[Any]: ...
def cancel_current_route(payment_state: Any, initiator_state: Any) -> Any: ...
def subdispatch_to_initiatortransfer(
    payment_state: Any,
    initiator_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def subdispatch_to_all_initiatortransfer(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def handle_block(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def handle_init(
    payment_state: Any,
    state_change: Any,
    addresses_to_channel: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def handle_cancelpayment(payment_state: Any, channelidentifiers_to_channels: Any) -> Any: ...
def handle_failroute(payment_state: Any, state_change: Any) -> Any: ...
def handle_transferreroute(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    addresses_to_channel: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def handle_lock_expired(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    block_number: Any,
) -> Any: ...
def handle_offchain_secretreveal(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def handle_onchain_secretreveal(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def handle_secretrequest(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
def state_transition(
    payment_state: Any,
    state_change: Any,
    channelidentifiers_to_channels: Any,
    addresses_to_channel: Any,
    pseudo_random_generator: Any,
    block_number: Any,
) -> Any: ...
```