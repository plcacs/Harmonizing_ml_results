from collections import Counter, defaultdict
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Set, Union
import pytest
from hypothesis import HealthCheck, Verbosity
from hypothesis.stateful import Bundle, RuleBasedStateMachine, Rule, consumes, initialize, invariant, multiple, rule
from hypothesis.strategies import binary, builds, composite, integers, random_module, randoms
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, DEFAULT_WAIT_BEFORE_LOCK_REMOVAL
from raiden.tests.fuzz.utils import SendLockedTransferInNode, SendSecretRequestInNode, SendSecretRevealInNode, SendUnlockInNode
from raiden.tests.utils.factories import make_block_hash
from raiden.transfer import channel, node
from raiden.transfer.events import EventInvalidReceivedLockedTransfer, EventPaymentReceivedSuccess, EventPaymentSentFailed, EventPaymentSentSuccess, SendProcessed
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed, EventUnlockClaimSuccess, EventUnlockSuccess, SendLockedTransfer, SendSecretRequest, SendSecretReveal, SendUnlock
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState
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
def payment_id(draw) -> int:
    ...

def event_types_match(events: List[Any], *expected_types: type) -> bool:
    ...

def transferred_amount(state: Any) -> TokenAmount:
    ...

@dataclass
class Route:
    channel_from: int
    hops: List[Address]
    initiator: Address
    target: Address

routes: Bundle[Route]
init_initiators: Bundle[Any]
init_mediators: Bundle[Any]
send_locked_transfers: Bundle[SendLockedTransferInNode]
secret_requests: Bundle[SendSecretRequestInNode]
send_secret_requests: Bundle[SendSecretRequestInNode]
send_secret_reveals_backward: Bundle[SendSecretRevealInNode]
send_secret_reveals_forward: Bundle[SendSecretRevealInNode]
send_unlocks: Bundle[SendUnlockInNode]

AddressToAmount = Dict[Address, TokenAmount]

def make_tokenamount_defaultdict() -> defaultdict[Address, TokenAmount]:
    ...

@dataclass
class TransferOrder:
    initiated: List[Any]
    answered: List[Any]

@dataclass
class Client:
    chain_state: ChainState
    address_to_channel: Dict[Address, NettingChannelState]
    expected_expiry: Dict[SecretHash, BlockNumber]
    our_previous_deposit: defaultdict[Address, TokenAmount]
    partner_previous_deposit: defaultdict[Address, TokenAmount]
    our_previous_transferred: defaultdict[Address, TokenAmount]
    partner_previous_transferred: defaultdict[Address, TokenAmount]
    our_previous_unclaimed: defaultdict[Address, TokenAmount]
    partner_previous_unclaimed: defaultdict[Address, TokenAmount]
    transfer_order: TransferOrder

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

    def new_channel(self, client_address: Address, partner_address: Address = None) -> Address:
        ...

    def _new_channel_transaction(self, client_address: Address, partner_address: Address) -> None:
        ...

    def new_channel_with_transaction(self, client_address: Address, partner_address: Address = None) -> Address:
        ...

    def new_client(self) -> Address:
        ...

    @initialize(target=Bundle[Route], block_number: int = ..., random: Random = ..., random_seed: Random = ...)
    def initialize_all(self, block_number: int, random: Random, random_seed: Random) -> multiple:
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

    def _action_init_initiator(self, route: Route, transfer: Any) -> Any:
        ...

    def _new_transfer_description(self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret) -> Any:
        ...

    @rule(target=send_locked_transfers, route: Route, payment_id: int, amount: int, secret: Secret)
    def valid_init_initiator(self, route: Route, payment_id: int, amount: int, secret: Secret) -> SendLockedTransferInNode:
        ...

    @rule(route: Route, payment_id: int, excess_amount: int, secret: Secret)
    def exceeded_capacity_init_initiator(self, route: Route, payment_id: int, excess_amount: int, secret: Secret) -> None:
        ...

    @rule(previous: SendLockedTransferInNode, route: Route, payment_id: int, amount: int)
    def used_secret_init_initiator(self, previous: SendLockedTransferInNode, route: Route, payment_id: int, amount: int) -> None:
        ...

    @rule(previous: SendLockedTransferInNode)
    def replay_init_initiator(self, previous: SendLockedTransferInNode) -> None:
        ...

    @rule(target=send_secret_reveals_forward, source: SendSecretRequestInNode)
    def process_valid_secret_request(self, source: SendSecretRequestInNode) -> multiple:
        ...

    @rule(source: SendSecretRequestInNode, wrong_amount: int)
    def process_secret_request_with_wrong_amount(self, source: SendSecretRequestInNode, wrong_amount: int) -> None:
        ...

    @rule(source: SendSecretRequestInNode, wrong_secret: Secret)
    def process_secret_request_with_wrong_secrethash(self, source: SendSecretRequestInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source: SendSecretRequestInNode, wrong_payment_identifier: int)
    def process_secret_request_with_wrong_payment_identifier(self, source: SendSecretRequestInNode, wrong_payment_identifier: int) -> None:
        ...

    @rule(target=send_unlocks, source: SendSecretRevealInNode)
    def process_secret_reveal_as_initiator(self, source: SendSecretRevealInNode) -> SendUnlockInNode:
        ...

    @rule(source: SendSecretRevealInNode, wrong_secret: Secret)
    def process_secret_reveal_with_mismatched_secret_as_initiator(self, source: SendSecretRevealInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source: SendSecretRevealInNode, wrong_secret: Secret)
    def process_secret_reveal_with_unknown_secrethash_as_initiator(self, source: SendSecretRevealInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source: SendSecretRevealInNode, wrong_channel_id: int)
    def process_secret_reveal_with_wrong_channel_identifier_as_initiator(self, source: SendSecretRevealInNode, wrong_channel_id: int) -> None:
        ...

    @rule(source: SendSecretRevealInNode, wrong_channel_id: int, wrong_recipient: Address)
    def process_secret_reveal_with_wrong_queue_identifier_as_initiator(self, source: SendSecretRevealInNode, wrong_channel_id: int, wrong_recipient: Address) -> None:
        ...

class TargetMixin:
    @rule(target=send_secret_requests, source: SendLockedTransferInNode)
    def process_send_locked_transfer(self, source: SendLockedTransferInNode) -> SendSecretRequestInNode:
        ...

    @rule(source: SendLockedTransferInNode, scrambling: Any)
    def process_send_locked_transfer_with_scrambled_balance_proof(self, source: SendLockedTransferInNode, scrambling: Any) -> None:
        ...

    @rule(source: SendLockedTransferInNode, scrambling: Any)
    def process_send_locked_transfer_with_scrambled_hash_time_lock_state(self, source: SendLockedTransferInNode, scrambling: Any) -> None:
        ...

    @rule(source: SendLockedTransferInNode, scrambling: Any)
    def process_send_locked_transfer_with_scrambled_locked_transfer_parameter(self, source: SendLockedTransferInNode, scrambling: Any) -> None:
        ...

    @rule(target=send_secret_reveals_backward, source: SendSecretRevealInNode)
    def process_secret_reveal_as_target(self, source: SendSecretRevealInNode) -> SendSecretRevealInNode:
        ...

    @rule(source: SendSecretRevealInNode, wrong_secret: Secret)
    def process_secret_reveal_with_mismatched_secret_as_target(self, source: SendSecretRevealInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source: SendSecretRevealInNode, wrong_secret: Secret)
    def process_secret_reveal_with_unknown_secrethash_as_target(self, source: SendSecretRevealInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source: SendSecretRevealInNode, wrong_channel_id: int)
    def process_secret_reveal_with_wrong_channel_identifier_as_target(self, source: SendSecretRevealInNode, wrong_channel_id: int) -> None:
        ...

    @rule(source: SendSecretRevealInNode, wrong_channel_id: int, wrong_recipient: Address)
    def process_secret_reveal_with_wrong_queue_identifier_as_target(self, source: SendSecretRevealInNode, wrong_channel_id: int, wrong_recipient: Address) -> None:
        ...

    @rule(source: SendUnlockInNode)
    def process_unlock(self, source: SendUnlockInNode) -> None:
        ...

    @rule(source: SendUnlockInNode, wrong_secret: Secret)
    def process_unlock_with_mismatched_secret(self, source: SendUnlockInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source: SendUnlockInNode, wrong_secret: Secret)
    def process_unlock_with_unknown_secrethash(self, source: SendUnlockInNode, wrong_secret: Secret) -> None:
        ...

    @rule(source: SendUnlockInNode, scrambling: Any)
    def process_unlock_with_scrambled_balance_proof(self, source: SendUnlockInNode, scrambling: Any) -> None:
        ...

class BalanceProofData:
    def __init__(self, canonical_identifier: Any) -> None:
        ...

    def update(self, amount: TokenAmount, lock: HashTimeLockState) -> None:
        ...

class WithOurAddress:
    def __init__(self, our_address: Address, data: Any) -> None:
        ...

class MediatorMixin:
    def __init__(self) -> None:
        ...

    def _get_balance_proof_data(self, partner: Address, client_address: Address) -> BalanceProofData:
        ...

    def _update_balance_proof_data(self, partner: Address, amount: TokenAmount, expiration: BlockExpiration, secret: Secret, our_address: Address) -> BalanceProofData:
        ...

    def _new_mediator_transfer(self, initiator_address: Address, target_address: Address, payment_id: int, amount: TokenAmount, secret: Secret, our_address: Address) -> LockedTransferSignedState:
        ...

    def _action_init_mediator(self, transfer: LockedTransferSignedState, client_address: Address) -> WithOurAddress:
        ...

    def _unwrap(self, with_our_address: WithOurAddress) -> tuple[Any, Client, Address]:
        ...

    @rule(target=init_mediators, payment_id: int, amount: int, secret: Secret)
    def valid_init_mediator(self, from_channel: WithOurAddress, to_channel: WithOurAddress, payment_id: int, amount: int, secret: Secret) -> WithOurAddress:
        ...

    @rule(target=secret_requests, previous_action_with_address: WithOurAddress)
    def valid_receive_secret_reveal(self, previous_action_with_address: WithOurAddress) -> WithOurAddress:
        ...

    @rule(previous_action_with_address: WithOurAddress)
    def replay_receive_secret_reveal(self, previous_action_with_address: WithOurAddress) -> None:
        ...

    @rule(previous_action_with_address: WithOurAddress, invalid_sender: Address)
    def replay_receive_secret_reveal_scrambled_sender(self, previous_action_with_address: WithOurAddress, invalid_sender: Address) -> None:
        ...

    @rule(previous_action_with_address: WithOurAddress, secret: Secret)
    def wrong_secret_receive_secret_reveal(self, previous_action_with_address: WithOurAddress, secret: Secret) -> None:
        ...

    @rule(target=secret_requests, previous_action_with_address: WithOurAddress, invalid_sender: Address)
    def wrong_address_receive_secret_reveal(self, previous_action_with_address: WithOurAddress, invalid_sender: Address) -> WithOurAddress:
        ...

class OnChainMixin:
    @rule(number: int)
    def new_blocks(self, number: int) -> None:
        ...

    @rule(reference: Route, target: Route)
    def open_channel(self, reference: Route) -> multiple:
        ...

    def routes_for_new_channel(self, from_address: Address, to_address: Address) -> multiple:
        ...

    @rule(reference: Route)
    def settle_channel(self, reference: Route) -> None:
        ...

class MediatorStateMachine(MediatorMixin, ChainStateStateMachine):
    ...

class OnChainStateMachine(OnChainMixin, ChainStateStateMachine):
    def create_network(self) -> multiple:
        ...

class MultiChannelMediatorStateMachine(MediatorMixin, OnChainMixin, ChainStateStateMachine):
    ...

class DirectTransfersStateMachine(InitiatorMixin, TargetMixin, ChainStateStateMachine):
    def create_network(self) -> multiple:
        ...

TestMediator = pytest.mark.skip(MediatorStateMachine.TestCase)
TestOnChain = OnChainStateMachine.TestCase
TestMultiChannelMediator = pytest.mark.skip(MultiChannelMediatorStateMachine.TestCase)
TestDirectTransfers = DirectTransfersStateMachine.TestCase

def unwrap_multiple(multiple_results: multiple) -> Any:
    ...

def test_regression_malicious_secret_request_handled_properly() -> None:
    ...