from collections import Counter, defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pytest
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, StatefulTestResult, multiple
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, DEFAULT_WAIT_BEFORE_LOCK_REMOVAL
from raiden.tests.fuzz.utils import SendLockedTransferInNode, SendSecretRevealInNode, SendUnlockInNode
from raiden.transfer import channel, node
from raiden.transfer.events import EventInvalidReceivedLockedTransfer, EventPaymentReceivedSuccess, EventPaymentSentFailed, EventPaymentSentSuccess, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed, EventUnlockClaimSuccess, EventUnlockSuccess, SendLockedTransfer, SendSecretRequest, SendSecretReveal, SendUnlock
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ActionInitMediator, ReceiveSecretReveal, TransferDescriptionWithSecretState
from raiden.transfer.state import ChainState, ChannelState, HashTimeLockState, NettingChannelState, TokenNetworkRegistryState, TokenNetworkState
from raiden.transfer.state_change import Block, ContractReceiveChannelNew, ContractReceiveChannelSettled
from raiden.utils.typing import Address, BlockExpiration, BlockGasLimit, BlockNumber, MessageID, Nonce, PrivateKey, Secret, SecretHash, TokenAddress, TokenAmount

@composite
def secret(draw) -> Secret:
    ...

@composite
def address(draw) -> Address:
    ...

@composite
def payment_id(draw) -> TokenAmount:
    ...

def event_types_match(events: List[Any], *expected_types: type) -> bool:
    ...

def transferred_amount(state: ChannelState) -> TokenAmount:
    ...

@dataclass
class Route:
    channel_from: int = 0
    hops: List[Address] = field(default_factory=list)

    @property
    def initiator(self) -> Address:
        ...

    @property
    def target(self) -> Address:
        ...

routes = Bundle('routes')
init_initiators = Bundle('init_initiators')
init_mediators = Bundle('init_mediators')
send_locked_transfers = Bundle('send_locked_transfers')
secret_requests = Bundle('secret_requests')
send_secret_requests = Bundle('send_secret_requests')
send_secret_reveals_backward = Bundle('send_secret_reveals_backward')
send_secret_reveals_forward = Bundle('send_secret_reveals_forward')
send_unlocks = Bundle('send_unlocks')

AddressToAmount = Dict[Address, TokenAmount]

def make_tokenamount_defaultdict() -> defaultdict[Address, TokenAmount]:
    ...

@dataclass
class TransferOrder:
    initiated: List[Any] = field(default_factory=list)
    answered: List[Any] = field(default_factory=list)

@dataclass
class Client:
    address_to_channel: Dict[Address, NettingChannelState] = field(default_factory=dict)
    expected_expiry: Dict[SecretHash, BlockNumber] = field(default_factory=dict)
    our_previous_deposit: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_deposit: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    our_previous_transferred: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_transferred: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    our_previous_unclaimed: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_unclaimed: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    transfer_order: TransferOrder = field(default_factory=TransferOrder)

    def assert_monotonicity_invariants(self) -> None:
        ...

    def assert_channel_state_invariants(self) -> None:
        ...

class ChainStateStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        ...

    def new_address(self) -> Address:
        ...

    def _new_channel_state(self, our_address: Address, partner_address: Address) -> NettingChannelState:
        ...

    def new_channel(self, client_address: Address, partner_address: Optional[Address] = None) -> Address:
        ...

    def _new_channel_transaction(self, client_address: Address, partner_address: Address) -> None:
        ...

    def new_channel_with_transaction(self, client_address: Address, partner_address: Optional[Address] = None) -> Address:
        ...

    def new_client(self) -> Address:
        ...

    @initialize(target=routes, block_number=st.integers(min_value=1), random=st.randoms(), random_seed=st.random_module())
    def initialize_all(self, block_number: BlockNumber, random: Random, random_seed: Random) -> multiple:
        ...

    def event(self, description: str) -> None:
        ...

    @invariant()
    def chain_state_invariants(self) -> None:
        ...

    def channel_opened(self, partner_address: Address, client_address: Address) -> bool:
        ...

    def create_network(self) -> multiple:
        ...

class InitiatorMixin:
    def __init__(self) -> None:
        ...

    def _available_amount(self, route: Route) -> TokenAmount:
        ...

    def _is_expired(self, secrethash: SecretHash, initiator: Address) -> bool:
        ...

    def _is_removed(self, action: Any) -> bool:
        ...

    def _action_init_initiator(self, route: Route, transfer: TransferDescriptionWithSecretState) -> ActionInitInitiator:
        ...

    def _new_transfer_description(self, route: Route, payment_id: TokenAmount, amount: TokenAmount, secret: Secret) -> TransferDescriptionWithSecretState:
        ...

    @rule(target=send_locked_transfers, route=routes, payment_id=payment_id(), amount=st.integers(min_value=1, max_value=100), secret=secret())
    def valid_init_initiator(self, route: Route, payment_id: TokenAmount, amount: TokenAmount, secret: Secret) -> SendLockedTransferInNode:
        ...

    @rule(route=routes, payment_id=payment_id(), excess_amount=st.integers(min_value=1), secret=secret())
    def exceeded_capacity_init_initiator(self, route: Route, payment_id: TokenAmount, excess_amount: int, secret: Secret) -> None:
        ...

    @rule(previous=send_locked_transfers, route=routes, payment_id=payment_id(), amount=st.integers(min_value=1))
    def used_secret_init_initiator(self, previous: SendLockedTransferInNode, route: Route, payment_id: TokenAmount, amount: TokenAmount) -> None:
        ...

    @rule(previous=send_locked_transfers)
    def replay_init_initiator(self, previous: SendLockedTransferInNode) -> None:
        ...

    @rule(target=send_secret_reveals_forward, source=consumes(send_secret_requests))
    def process_valid_secret_request(self, source: SendSecretRequestInNode) -> Union[multiple, SendSecretRevealInNode]:
        ...

    @rule(source=send_secret_requests, wrong_amount=st.integers())
    def process_secret_request_with_wrong_amount(self, source: SendSecretRequestInNode, wrong_amount: int) -> None:
        ...

    @rule(source=send_secret_requests, wrong_secret=secret())
    def process_secret_request_with_wrong_secrethash(self, source: SendSecretRequestInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source=send_secret_requests, wrong_payment_identifier=st.integers())
    def process_secret_request_with_wrong_payment_identifier(self, source: SendSecretRequestInNode, wrong_payment_identifier: int) -> None:
        ...

    @rule(target=send_unlocks, source=consumes(send_secret_reveals_backward))
    def process_secret_reveal_as_initiator(self, source: SendSecretRevealInNode) -> SendUnlockInNode:
        ...

    @rule(source=send_secret_reveals_backward, wrong_secret=secret())
    def process_secret_reveal_with_mismatched_secret_as_initiator(self, source: SendSecretRevealInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source=send_secret_reveals_backward, wrong_secret=secret())
    def process_secret_reveal_with_unknown_secrethash_as_initiator(self, source: SendSecretRevealInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source=send_secret_reveals_backward, wrong_channel_id=st.integers())
    def process_secret_reveal_with_wrong_channel_identifier_as_initiator(self, source: SendSecretRevealInNode, wrong_channel_id: int) -> None:
        ...

    @rule(source=send_secret_reveals_backward, wrong_channel_id=st.integers(), wrong_recipient=address())
    def process_secret_reveal_with_wrong_queue_identifier_as_initiator(self, source: SendSecretRevealInNode, wrong_channel_id: int, wrong_recipient: Address) -> None:
        ...