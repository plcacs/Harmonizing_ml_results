from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Dict, List, Set, Optional, Tuple, TypeVar, Generic, Union
import pytest
from hypothesis import assume, event
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, initialize, invariant, multiple, rule
from hypothesis.strategies import binary, builds, composite, integers, random_module, randoms
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

T = TypeVar('T')

@composite
def secret(draw: Any) -> Secret:
    return draw(builds(random_secret))

@composite
def address(draw: Any) -> Address:
    return draw(binary(min_size=20, max_size=20))

@composite
def payment_id(draw: Any) -> int:
    return draw(integers(min_value=1, max_value=UINT64_MAX))

def event_types_match(events: List[Any], *expected_types: Any) -> bool:
    return Counter([type(event) for event in events]) == Counter(expected_types)

def transferred_amount(state: Any) -> TokenAmount:
    return TokenAmount(0) if not state.balance_proof else state.balance_proof.transferred_amount

@dataclass
class Route:
    channel_from: int = 0
    hops: Tuple[Address, ...] = field(default_factory=tuple)

    @property
    def initiator(self) -> Address:
        return self.hops[0]

    @property
    def target(self) -> Address:
        return self.hops[-1]

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

def make_tokenamount_defaultdict() -> defaultdict:
    return defaultdict(lambda: TokenAmount(0))

@dataclass
class TransferOrder:
    initiated: List[Secret] = field(default_factory=list)
    answered: List[Secret] = field(default_factory=list)

@dataclass
class Client:
    address_to_channel: Dict[Address, NettingChannelState] = field(default_factory=dict)
    expected_expiry: Dict[SecretHash, BlockNumber] = field(default_factory=dict)
    our_previous_deposit: defaultdict = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_deposit: defaultdict = field(default_factory=make_tokenamount_defaultdict)
    our_previous_transferred: defaultdict = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_transferred: defaultdict = field(default_factory=make_tokenamount_defaultdict)
    our_previous_unclaimed: defaultdict = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_unclaimed: defaultdict = field(default_factory=make_tokenamount_defaultdict)
    transfer_order: TransferOrder = field(default_factory=TransferOrder)
    chain_state: Optional[ChainState] = None

    def assert_monotonicity_invariants(self) -> None:
        """Assert all monotonicity properties stated in Raiden specification"""
        for address, netting_channel in self.address_to_channel.items():
            assert netting_channel.our_total_deposit >= self.our_previous_deposit[address]
            assert netting_channel.partner_total_deposit >= self.partner_previous_deposit[address]
            self.our_previous_deposit[address] = netting_channel.our_total_deposit
            self.partner_previous_deposit[address] = netting_channel.partner_total_deposit
            our_transferred = transferred_amount(netting_channel.our_state)
            partner_transferred = transferred_amount(netting_channel.partner_state)
            our_unclaimed = channel.get_amount_unclaimed_onchain(netting_channel.our_state)
            partner_unclaimed = channel.get_amount_unclaimed_onchain(netting_channel.partner_state)
            assert our_transferred >= self.our_previous_transferred[address]
            assert partner_transferred >= self.partner_previous_transferred[address]
            assert our_unclaimed + our_transferred >= self.our_previous_transferred[address] + self.our_previous_unclaimed[address]
            assert partner_unclaimed + partner_transferred >= self.partner_previous_transferred[address] + self.partner_previous_unclaimed[address]
            self.our_previous_transferred[address] = our_transferred
            self.partner_previous_transferred[address] = partner_transferred
            self.our_previous_unclaimed[address] = our_unclaimed
            self.partner_previous_unclaimed[address] = partner_unclaimed

    def assert_channel_state_invariants(self) -> None:
        """Assert all channel state invariants given in the Raiden specification"""
        for netting_channel in self.address_to_channel.values():
            our_state = netting_channel.our_state
            partner_state = netting_channel.partner_state
            our_transferred_amount = TokenAmount(0)
            if our_state.balance_proof:
                our_transferred_amount = our_state.balance_proof.transferred_amount
                assert our_transferred_amount >= TokenAmount(0)
            partner_transferred_amount = TokenAmount(0)
            if partner_state.balance_proof:
                partner_transferred_amount = partner_state.balance_proof.transferred_amount
                assert partner_transferred_amount >= TokenAmount(0)
            assert channel.get_distributable(our_state, partner_state) >= TokenAmount(0)
            assert channel.get_distributable(partner_state, our_state) >= TokenAmount(0)
            our_deposit = netting_channel.our_total_deposit
            partner_deposit = netting_channel.partner_total_deposit
            total_deposit = our_deposit + partner_deposit
            our_amount_locked = channel.get_amount_locked(our_state)
            our_balance = channel.get_balance(our_state, partner_state)
            partner_amount_locked = channel.get_amount_locked(partner_state)
            partner_balance = channel.get_balance(partner_state, our_state)
            assert TokenAmount(0) <= our_amount_locked <= our_balance
            assert TokenAmount(0) <= partner_amount_locked <= partner_balance
            assert our_amount_locked <= total_deposit
            assert partner_amount_locked <= total_deposit
            our_transferred = partner_transferred_amount - our_transferred_amount
            netted_transferred = our_transferred + partner_amount_locked - our_amount_locked
            assert TokenAmount(0) <= our_deposit + our_transferred - our_amount_locked <= total_deposit
            assert TokenAmount(0) <= partner_deposit - our_transferred - partner_amount_locked <= total_deposit
            assert -our_deposit <= netted_transferred <= partner_deposit

class ChainStateStateMachine(RuleBasedStateMachine):

    def __init__(self) -> None:
        self.replay_path = False
        self.address_to_privkey: Dict[Address, PrivateKey] = {}
        self.address_to_client: Dict[Address, Client] = {}
        self.transfer_order = TransferOrder()
        super().__init__()

    def new_address(self) -> Address:
        privkey, address = factories.make_privkey_address()
        self.address_to_privkey[address] = privkey
        return address

    def _new_channel_state(self, our_address: Address, partner_address: Address) -> NettingChannelState:
        identifier = factories.make_canonical_identifier(token_network_address=self.token_network_address)
        our_state = factories.NettingChannelEndStateProperties(balance=TokenAmount(1000), address=our_address)
        partner_state = factories.NettingChannelEndStateProperties(balance=TokenAmount(1000), address=partner_address)
        return factories.create(factories.NettingChannelStateProperties(our_state=our_state, partner_state=partner_state, canonical_identifier=identifier))

    def new_channel(self, client_address: Address, partner_address: Optional[Address] = None) -> Address:
        """Create a new partner address with private key and channel. The
        private key and channels are listed in the instance's dictionaries,
        the address is returned and should be added to the partners Bundle.
        """
        if not partner_address:
            partner_address = self.new_address()
        client = self.address_to_client[client_address]
        channel = self._new_channel_state(client_address, partner_address)
        client.address_to_channel[partner_address] = channel
        partner_client = self.address_to_client.get(partner_address)
        if partner_client is not None:
            mirrored = deepcopy(channel)
            mirrored.our_state, mirrored.partner_state = (mirrored.partner_state, mirrored.our_state)
            partner_client.address_to_channel[client_address] = mirrored
        return partner_address

    def _new_channel_transaction(self, client_address: Address, partner_address: Address) -> None:
        client = self.address_to_client[client_address]
        channel_state = client.address_to_channel[partner_address]
        assert isinstance(channel_state, NettingChannelState)
        channel_new_state_change = ContractReceiveChannelNew(transaction_hash=factories.make_transaction_hash(), channel_state=channel_state, block_number=self.block_number, block_hash=factories.make_block_hash())
        node.state_transition(client.chain_state, channel_new_state_change)

    def new_channel_with_transaction(self, client_address: Address, partner_address: Optional[Address] = None) -> Address:
        partner_address = self.new_channel(client_address, partner_address)
        self._new_channel_transaction(client_address, partner_address)
        if partner_address in self.address_to_client:
            self._new_channel_transaction(partner_address, client_address)
        return partner_address

    def new_client(self) -> Address:
        address = self.new_address()
        chain_state = ChainState(pseudo_random_generator=self.random, block_number=self.block_number, block_hash=self.block_hash, our_address=address, chain_id=factories.UNIT_CHAIN_ID)
        chain_state.identifiers_to_tokennetworkregistries[self.token_network_registry_address] = deepcopy(self.token_network_registry_state)
        chain_state.tokennetworkaddresses_to_tokennetworkregistryaddresses[self.token_network_address] = self.token_network_registry_address
        self.address_to_client[address] = Client(chain_state=chain_state)
        return address

    @initialize(target=routes, block_number=integers(min_value=GENESIS_BLOCK_NUMBER + 1), random=randoms(), random_seed=random_module())
    def initialize_all(self, block_number: BlockNumber, random: Random, random_seed: Any) -> multiple:
        self.random_seed = random_seed
        self.block_number = block_number
        self.block_hash = factories.make_block_hash()
        self.random = random
        self.token_network_address = factories.UNIT_TOKEN_NETWORK_ADDRESS
        self.token_id = factories.UNIT_TOKEN_ADDRESS
        self.token_network_state = TokenNetworkState(address=self.token_network_address, token_address=self.token_id)
        self.token_network_registry_address = factories.make_token_network_registry_address()
        self.token_network_registry_state = TokenNetworkRegistryState(self.token_network_registry_address, [self.token_network_state])
        return multiple(*self.create_network())

    def event(self, description: str) -> None:
        """Wrapper for hypothesis' event function.

        hypothesis.event raises an exception when invoked outside of hypothesis
        context, so skip it when we are replaying a failed path.
        """
        if not self.replay_path:
            event(description)

    @invariant()
    def chain_state_invariants(self) -> None:
        for client in self.address_to_client.values():
            client.assert_monotonicity_invariants()
            client.assert_channel_state_invariants()

    def channel_opened(self, partner_address: Address, client_address: Address) -> bool:
        try:
            client = self.address_to_client[client_address]
        except KeyError:
            return False
        else:
            needed_channel = client.address_to_channel[partner_address]
            return channel.get_status(needed_channel) == ChannelState.STATE_OPENED

    def create_network(self) -> List[Route]:
        raise NotImplementedError('Every fuzz test needs to override this.')

class InitiatorMixin:

    def __init__(self) -> None:
        super().__init__()
        self.used_secrets: Set[Secret] = set()
        self.processed_secret_request_secrethashes: Set[SecretHash] = set()
        self.initiated: Set[Secret] = set()

    def _available_amount(self, route: Route) -> TokenAmount:
        client = self.address_to_client[route.initiator]
        netting_channel = client.address_to_channel[route.hops[1]]
        return channel.get_distributable(netting_channel.our_state, netting_channel.partner_state)

    def _is_expired(self, secrethash: SecretHash, initiator: Address) -> bool:
        expiry = self.address_to_client[initiator].expected_expiry[secrethash]
        return self.block_number >= expiry + DEFAULT_WAIT_BEFORE_LOCK_REMOVAL

    def _is_removed(self, action: ActionInitInitiator) -> bool:
        return self._is_expired(action.transfer.secrethash, action.transfer.initiator)

    def _action_init_initiator(self, route: Route, transfer: TransferDescriptionWithSecretState) -> ActionInitInitiator:
        client = self.address_to_client[route.initiator]
        channel = client.address_to_channel[route.hops[1]]
        if transfer.secrethash not in client.expected_expiry:
            client.expected_expiry[transfer.secrethash] = self.block_number + 10
        return ActionInitInitiator(transfer, [factories.make_route_from_channel(channel)])

    def _new_transfer_description(self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret) -> TransferDescriptionWithSecretState:
        self.used_secrets.add(secret)
        return TransferDescriptionWithSecretState(token_network_registry_address=self.token_network_registry_address, payment_identifier=payment_id, amount=amount, token_network_address=self.token_network_address, initiator=route.initiator, target=route.target, secret=secret)

    @rule(target=send_locked_transfers, route=routes, payment_id=payment_id(), amount=integers(min_value=1, max_value=100), secret=secret())
    def valid_init_initiator(self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret) -> utils.SendLockedTransferInNode:
        assume(amount <= self._available_amount(route))
        assume(secret not in self.used_secrets)
        transfer = self._new_transfer_description(route, payment_id, amount, secret)
        action = self._action_init_initiator(route, transfer)
        client = self.address_to_client[route.initiator]
        result = node.state_transition(client.chain_state, action)
        assert event_types_match(result.events, SendLockedTransfer)
        self.initiated.add(transfer.secret)
        client.expected_expiry[transfer.secrethash] = self.block_number + 10
        self.transfer_order.initiated.append(secret)
        return utils.SendLockedTransferInNode(event=result.events[0], action=action, node=route.initiator, private_key=self.address_to_privkey[route.initiator])

    @rule(route=routes, payment_id=payment_id(), excess_amount=integers(min_value=1), secret=secret())
    def exceeded_capacity_init_initiator(self, route: Route, payment_id: int, excess_amount: TokenAmount, secret: Secret) -> None:
        amount = self._available_amount(route) + excess_amount
        transfer = self._new_transfer_description(route, payment_id, amount, secret)
        action = self._action_init_initiator(route, transfer)
        client = self.address_to_client[route.initiator]
        result = node.state_transition(client.chain_state, action)
        assert event_types_match(result.events, EventPaymentSentFailed)
        self.event('ActionInitInitiator failed: Amount exceeded')

    @rule(previous=send_locked_transfers, route=routes, payment_id=payment_id(), amount=integers(min_value=1))
    def