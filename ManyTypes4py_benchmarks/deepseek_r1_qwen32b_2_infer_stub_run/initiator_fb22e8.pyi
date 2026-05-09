import random
from math import ceil
from typing import Dict, List, Optional, Tuple, Union
from raiden.settings import (
    DEFAULT_MEDIATION_FEE_MARGIN,
    DEFAULT_WAIT_BEFORE_LOCK_REMOVAL,
    MAX_MEDIATION_FEE_PERC,
    PAYMENT_AMOUNT_BASED_FEE_MARGIN,
)
from raiden.utils.typing import (
    Address,
    BlockExpiration,
    BlockNumber,
    FeeAmount,
    MessageID,
    MYPY_ANNOTATION,
    Optional,
    PaymentAmount,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    TokenNetworkAddress,
)
from raiden.transfer import channel
from raiden.transfer.architecture import Event, TransitionResult
from raiden.transfer.events import (
    EventInvalidSecretRequest,
    EventPaymentSentFailed,
    EventPaymentSentSuccess,
)
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE
from raiden.transfer.mediated_transfer.events import (
    EventRouteFailed,
    EventUnlockFailed,
    EventUnlockSuccess,
    SendLockedTransfer,
    SendSecretReveal,
)
from raiden.transfer.mediated_transfer.state import InitiatorTransferState
from raiden.transfer.mediated_transfer.state_change import (
    ReceiveSecretRequest,
    ReceiveSecretReveal,
)
from raiden.transfer.state import ChannelState, NettingChannelState, RouteState
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, StateChange

def calculate_fee_margin(payment_amount: PaymentAmount, estimated_fee: FeeAmount) -> FeeAmount:
    ...

def calculate_safe_amount_with_fee(payment_amount: PaymentAmount, estimated_fee: FeeAmount) -> PaymentWithFeeAmount:
    ...

def events_for_unlock_lock(
    initiator_state: InitiatorTransferState,
    channel_state: ChannelState,
    secret: Secret,
    secrethash: SecretHash,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> List[Union[EventUnlock, EventPaymentSentSuccess, EventUnlockSuccess]]:
    ...

def handle_block(
    initiator_state: InitiatorTransferState,
    state_change: Block,
    channel_state: ChannelState,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    ...

def try_new_route(
    addresses_to_channel: Dict[Tuple[TokenNetworkAddress, Address], ChannelState],
    candidate_route_states: List[RouteState],
    transfer_description: TransferDescriptionWithSecretState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult:
    ...

def send_lockedtransfer(
    transfer_description: TransferDescriptionWithSecretState,
    channel_state: ChannelState,
    message_identifier: MessageID,
    block_number: BlockNumber,
    route_state: RouteState,
    route_states: List[RouteState],
) -> SendLockedTransfer:
    ...

def handle_secretrequest(
    initiator_state: InitiatorTransferState,
    state_change: ReceiveSecretRequest,
    channel_state: ChannelState,
    pseudo_random_generator: random.Random,
) -> TransitionResult:
    ...

def handle_offchain_secretreveal(
    initiator_state: InitiatorTransferState,
    state_change: ReceiveSecretReveal,
    channel_state: ChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult:
    ...

def handle_onchain_secretreveal(
    initiator_state: InitiatorTransferState,
    state_change: ContractReceiveSecretReveal,
    channel_state: ChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult:
    ...

def state_transition(
    initiator_state: InitiatorTransferState,
    state_change: StateChange,
    channel_state: ChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult:
    ...