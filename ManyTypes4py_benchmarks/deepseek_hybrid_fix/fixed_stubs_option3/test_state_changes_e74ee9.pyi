from collections import Counter, defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Set, Callable, Counter as CounterType, Optional, Tuple, Type, TypeVar, Union
import pytest
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, initialize, invariant, multiple, rule
from hypothesis.strategies import binary, builds, composite, integers, random_module, randoms, DrawFn, SearchStrategy
from raiden.constants import GENESIS_BLOCK_NUMBER, LOCKSROOT_OF_NO_LOCKS, UINT64_MAX
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, DEFAULT_WAIT_BEFORE_LOCK_REMOVAL
from raiden.tests.fuzz import utils
from raiden.tests.utils import factories
from raiden.tests.utils.factories import make_block_hash
from raiden.transfer import channel, node
from raiden.transfer.architecture import StateChange
from raiden.transfer.channel import compute_locksroot
from raiden.transfer.events import EventInvalidReceivedLockedTransfer, EventPaymentReceivedSuccess, EventPaymentSentFailed, EventPaymentSentSuccess, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed, EventUnlockClaimSuccess, EventUnlockSuccess, SendLockedTransfer, SendSecretRequest, SendSecretReveal, SendUnlock
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ActionInitMediator, ReceiveSecretReveal, TransferDescriptionWithSecretState
from raiden.transfer.state import ChainState, ChannelState, HashTimeLockState, NettingChannelState, TokenNetworkRegistryState, TokenNetworkState, make_empty_pending_locks_state
from raiden.transfer.state_change import Block, ContractReceiveChannelNew, ContractReceiveChannelSettled
from raiden.utils.copy import deepcopy
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.transfers import random_secret
from raiden.utils.typing import Address, BlockExpiration, BlockGasLimit, BlockNumber, MessageID, Nonce, PrivateKey, Secret, SecretHash, TokenAddress, TokenAmount

@composite
def secret(draw: DrawFn) -> Secret: ...

@composite
def address(draw: DrawFn) -> Address: ...

@composite
def payment_id(draw: DrawFn) -> int: ...

def event_types_match(events: List[Any], *expected_types: Type[Any]) -> bool: ...

def transferred_amount(state: Any) -> TokenAmount: ...

@dataclass
class Route:
    channel_from: int = 0
    hops: Tuple[Address, ...] = ()

    @property
    def initiator(self) -> Address: ...
    @property
    def target(self) -> Address: ...

routes: Bundle = Bundle('routes')
init_initiators: Bundle = Bundle('init_initiators')
init_mediators: Bundle = Bundle('init_mediators')
send_locked_transfers: Bundle = Bundle('send_locked_transfers')
secret_requests: Bundle = Bundle('secret_requests')
send_secret_requests: Bundle = Bundle('send_secret_requests')
send_secret_reveals_backward: Bundle = Bundle('send_secret_reveals_backward')
send_secret_reveals_forward: Bundle = Bundle('send_secret_reveals_forward')
send_unlocks: Bundle = Bundle('send_unlocks')
AddressToAmount = Dict[Address, TokenAmount]

def make_tokenamount_defaultdict() -> defaultdict[Address, TokenAmount]: ...

@dataclass
class TransferOrder:
    initiated: List[Any] = field(default_factory=list)
    answered: List[Any] = field(default_factory=list)

@dataclass
class Client:
    chain_state: ChainState
    address_to_channel: Dict[Address, NettingChannelState] = field(default_factory=dict)
    expected_expiry: Dict[SecretHash, BlockNumber] = field(default_factory=dict)
    our_previous_deposit: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_deposit: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    our_previous_transferred: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_transferred: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    our_previous_unclaimed: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_unclaimed: defaultdict[Address, TokenAmount] = field(default_factory=make_tokenamount_defaultdict)
    transfer_order: TransferOrder = field(default_factory=TransferOrder)

    def assert_monotonicity_invariants(self) -> None: ...
    def assert_channel_state_invariants(self) -> None: ...

class ChainStateStateMachine(RuleBasedStateMachine):
    replay_path: bool
    address_to_privkey: Dict[Address, PrivateKey]
    address_to_client: Dict[Address, Client]
    transfer_order: TransferOrder
    token_network_address: Address
    token_id: TokenAddress
    token_network_state: TokenNetworkState
    token_network_registry_address: Address
    token_network_registry_state: TokenNetworkRegistryState
    block_number: BlockNumber
    block_hash: bytes
    random: Random
    random_seed: Any

    def __init__(self) -> None: ...
    def new_address(self) -> Address: ...
    def _new_channel_state(self, our_address: Address, partner_address: Address) -> NettingChannelState: ...
    def new_channel(self, client_address: Address, partner_address: Optional[Address] = None) -> Address: ...
    def _new_channel_transaction(self, client_address: Address, partner_address: Address) -> None: ...
    def new_channel_with_transaction(self, client_address: Address, partner_address: Optional[Address] = None) -> Address: ...
    def new_client(self) -> Address: ...
    @initialize(target=routes, block_number=integers(min_value=GENESIS_BLOCK_NUMBER + 1), random=randoms(), random_seed=random_module())
    def initialize_all(self, block_number: BlockNumber, random: Random, random_seed: Any) -> multiple: ...
    def event(self, description: str) -> None: ...
    @invariant()
    def chain_state_invariants(self) -> None: ...
    def channel_opened(self, partner_address: Address, client_address: Address) -> bool: ...
    def create_network(self) -> List[Route]: ...

class InitiatorMixin:
    used_secrets: Set[Secret]
    processed_secret_request_secrethashes: Set[SecretHash]
    initiated: Set[Secret]
    address_to_client: Dict[Address, Client]
    address_to_privkey: Dict[Address, PrivateKey]
    block_number: BlockNumber
    transfer_order: TransferOrder
    token_network_registry_address: Address
    token_network_address: Address

    def __init__(self) -> None: ...
    def _available_amount(self, route: Route) -> TokenAmount: ...
    def _is_expired(self, secrethash: SecretHash, initiator: Address) -> bool: ...
    def _is_removed(self, action: ActionInitInitiator) -> bool: ...
    def _action_init_initiator(self, route: Route, transfer: TransferDescriptionWithSecretState) -> ActionInitInitiator: ...
    def _new_transfer_description(self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret) -> TransferDescriptionWithSecretState: ...
    @rule(target=send_locked_transfers, route=routes, payment_id=payment_id(), amount=integers(min_value=1, max_value=100), secret=secret())
    def valid_init_initiator(self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret) -> Any: ...
    @rule(route=routes, payment_id=payment_id(), excess_amount=integers(min_value=1), secret=secret())
    def exceeded_capacity_init_initiator(self, route: Route, payment_id: int, excess_amount: TokenAmount, secret: Secret) -> None: ...
    @rule(previous=send_locked_transfers, route=routes, payment_id=payment_id(), amount=integers(min_value=1))
    def used_secret_init_initiator(self, previous: Any, route: Route, payment_id: int, amount: TokenAmount) -> None: ...
    @rule(previous=send_locked_transfers)
    def replay_init_initiator(self, previous: Any) -> None: ...
    @rule(target=send_secret_reveals_forward, source=consumes(send_secret_requests))
    def process_valid_secret_request(self, source: Any) -> multiple: ...
    @rule(source=send_secret_requests, wrong_amount=integers())
    def process_secret_request_with_wrong_amount(self, source: Any, wrong_amount: TokenAmount) -> None: ...
    @rule(source=send_secret_requests, wrong_secret=secret())
    def process_secret_request_with_wrong_secrethash(self, source: Any, wrong_secret: Secret) -> None: ...
    @rule(source=send_secret_requests, wrong_payment_identifier=integers())
    def process_secret_request_with_wrong_payment_identifier(self, source: Any, wrong_payment_identifier: int) -> None: ...
    @rule(target=send_unlocks, source=consumes(send_secret_reveals_backward))
    def process_secret_reveal_as_initiator(self, source: Any) -> Any: ...
    @rule(source=send_secret_reveals_backward, wrong_secret=secret())
    def process_secret_reveal_with_mismatched_secret_as_initiator(self, source: Any, wrong_secret: Secret) -> None: ...
    @rule(source=send_secret_reveals_backward, wrong_secret=secret())
    def process_secret_reveal_with_unknown_secrethash_as_initiator(self, source: Any, wrong_secret: Secret) -> None: ...
    @rule(source=send_secret_reveals_backward, wrong_channel_id=integers())
    def process_secret_reveal_with_wrong_channel_identifier_as_initiator(self, source: Any, wrong_channel_id: int) -> None: ...
    @rule(source=send_secret_reveals_backward, wrong_channel_id=integers(), wrong_recipient=address())
    def process_secret_reveal_with_wrong_queue_identifier_as_initiator(self, source: Any, wrong_channel_id: int, wrong_recipient: Address) -> None: ...

class TargetMixin:
    address_to_client: Dict[Address, Client]
    transfer_order: TransferOrder
    replay_path: bool

    @rule(target=send_secret_requests, source=consumes(send_locked_transfers))
    def process_send_locked_transfer(self, source: Any) -> Any: ...
    @rule(source=send_locked_transfers, scrambling=utils.balance_proof_scrambling())
    def process_send_locked_transfer_with_scrambled_balance_proof(self, source: Any, scrambling: Any) -> None: ...
    @rule(source=send_locked_transfers, scrambling=utils.hash_time_lock_scrambling())
    def process_send_locked_transfer_with_scrambled_hash_time_lock_state(self, source: Any, scrambling: Any) -> None: ...
    @rule(source=send_locked_transfers, scrambling=utils.locked_transfer_scrambling())
    def process_send_locked_transfer_with_scrambled_locked_transfer_parameter(self, source: Any, scrambling: Any) -> None: ...
    @rule(target=send_secret_reveals_backward, source=consumes(send_secret_reveals_forward))
    def process_secret_reveal_as_target(self, source: Any) -> Any: ...
    @rule(source=send_secret_reveals_forward, wrong_secret=secret())
    def process_secret_reveal_with_mismatched_secret_as_target(self, source: Any, wrong_secret: Secret) -> None: ...
    @rule(source=send_secret_reveals_forward, wrong_secret=secret())
    def process_secret_reveal_with_unknown_secrethash_as_target(self, source: Any, wrong_secret: Secret) -> None: ...
    @rule(source=send_secret_reveals_forward, wrong_channel_id=integers())
    def process_secret_reveal_with_wrong_channel_identifier_as_target(self, source: Any, wrong_channel_id: int) -> None: ...
    @rule(source=send_secret_reveals_forward, wrong_channel_id=integers(), wrong_recipient=address())
    def process_secret_reveal_with_wrong_queue_identifier_as_target(self, source: Any, wrong_channel_id: int, wrong_recipient: Address) -> None: ...
    @rule(source=consumes(send_unlocks))
    def process_unlock(self, source: Any) -> None: ...
    @rule(source=send_unlocks, wrong_secret=secret())
    def process_unlock_with_mismatched_secret(self, source: Any, wrong_secret: Secret) -> None: ...
    @rule(source=send_unlocks, wrong_secret=secret())
    def process_unlock_with_unknown_secrethash(self, source: Any, wrong_secret: Secret) -> None: ...
    @rule(source=send_unlocks, scrambling=utils.balance_proof_scrambling())
    def process_unlock_with_scrambled_balance_proof(self, source: Any, scrambling: Any) -> None: ...

class BalanceProofData:
    def __init__(self, canonical_identifier: Any) -> None: ...
    def update(self, amount: TokenAmount, lock: HashTimeLockState) -> None: ...

@dataclass
class WithOurAddress:
    our_address: Address
    data: Any

class MediatorMixin:
    partner_to_balance_proof_data: Dict[Address, BalanceProofData]
    secrethash_to_secret: Dict[SecretHash, Secret]
    waiting_for_unlock: Dict[Secret, Address]
    address_to_client: Dict[Address, Client]
    address_to_privkey: Dict[Address, PrivateKey]
    block_number: BlockNumber
    token_id: TokenAddress
    initial_number_of_channels: int

    def __init__(self) -> None: ...
    def _get_balance_proof_data(self, partner: Address, client_address: Address) -> BalanceProofData: ...
    def _update_balance_proof_data(self, partner: Address, amount: TokenAmount, expiration: BlockExpiration, secret: Secret, our_address: Address) -> BalanceProofData: ...
    def _new_mediator_transfer(self, initiator_address: Address, target_address: Address, payment_id: int, amount: TokenAmount, secret: Secret, our_address: Address) -> LockedTransferSignedState: ...
    def _action_init_mediator(self, transfer: LockedTransferSignedState, client_address: Address) -> WithOurAddress: ...
    def _unwrap(self, with_our_address: WithOurAddress) -> Tuple[Any, Client, Address]: ...
    @rule(target=init_mediators, payment_id=payment_id(), amount=integers(min_value=1, max_value=100), secret=secret())
    def valid_init_mediator(self, from_channel: Any, to_channel: Any, payment_id: int, amount: TokenAmount, secret: Secret) -> WithOurAddress: ...
    @rule(target=secret_requests, previous_action_with_address=consumes(init_mediators))
    def valid_receive_secret_reveal(self, previous_action_with_address: WithOurAddress) -> WithOurAddress: ...
    @rule(previous_action_with_address=secret_requests)
    def replay_receive_secret_reveal(self, previous_action_with_address: WithOurAddress) -> None: ...
    @rule(previous_action_with_address=secret_requests, invalid_sender=address())
    def replay_receive_secret_reveal_scrambled_sender(self, previous_action_with_address: WithOurAddress, invalid_sender: Address) -> None: ...
    @rule(previous_action_with_address=init_mediators, secret=secret())
    def wrong_secret_receive_secret_reveal(self, previous_action_with_address: WithOurAddress, secret: Secret) -> None: ...
    @rule(target=secret_requests, previous_action_with_address=consumes(init_mediators), invalid_sender=address())
    def wrong_address_receive_secret_reveal(self, previous_action_with_address: WithOurAddress, invalid_sender: Address) -> WithOurAddress: ...

class OnChainMixin:
    block_number: BlockNumber
    address_to_client: Dict[Address, Client]

    @rule(number=integers(min_value=1, max_value=50))
    def new_blocks(self, number: int) -> None: ...
    @rule(reference=routes, target=routes)
    def open_channel(self, reference: Route) -> multiple: ...
    def routes_for_new_channel(self, from_address: Address, to_address: Address) -> multiple: ...
    @rule(reference=consumes(routes))
    def settle_channel(self, reference: Route) -> None: ...

class MediatorStateMachine(MediatorMixin, ChainStateStateMachine):
    pass

class OnChainStateMachine(OnChainMixin, ChainStateStateMachine):
    def create_network(self) -> List[Route]: ...

class MultiChannelMediatorStateMachine(MediatorMixin, OnChainMixin, ChainStateStateMachine):
    pass

class DirectTransfersStateMachine(InitiatorMixin, TargetMixin, ChainStateStateMachine):
    def create_network(self) -> List[Route]: ...

TestMediator = pytest.mark.skip(MediatorStateMachine.TestCase)
TestOnChain = OnChainStateMachine.TestCase
TestMultiChannelMediator = pytest.mark.skip(MultiChannelMediatorStateMachine.TestCase)
TestDirectTransfers = DirectTransfersStateMachine.TestCase

def unwrap_multiple(multiple_results: multiple) -> Any: ...

def test_regression_malicious_secret_request_handled_properly() -> None: ...