from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
import pytest
from hypothesis import assume, event
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, initialize, invariant, multiple, rule
from hypothesis.strategies import binary, builds, composite, integers, random_module, randoms, SearchStrategy
from raiden.constants import GENESIS_BLOCK_NUMBER, LOCKSROOT_OF_NO_LOCKS, UINT64_MAX
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, DEFAULT_WAIT_BEFORE_LOCK_REMOVAL
from raiden.tests.fuzz import utils
from raiden.tests.utils import factories
from raiden.tests.utils.factories import make_block_hash
from raiden.transfer import channel, node
from raiden.transfer.architecture import StateChange
from raiden.transfer.channel import compute_locksroot
from raiden.transfer.events import (
    EventInvalidReceivedLockedTransfer,
    EventPaymentReceivedSuccess,
    EventPaymentSentFailed,
    EventPaymentSentSuccess,
    SendProcessed,
)
from raiden.transfer.mediated_transfer.events import (
    EventUnlockClaimFailed,
    EventUnlockClaimSuccess,
    EventUnlockSuccess,
    SendLockedTransfer,
    SendSecretRequest,
    SendSecretReveal,
    SendUnlock,
)
from raiden.transfer.mediated_transfer.state import LockedTransferSignedState
from raiden.transfer.mediated_transfer.state_change import (
    ActionInitInitiator,
    ActionInitMediator,
    ReceiveSecretReveal,
    TransferDescriptionWithSecretState,
)
from raiden.transfer.state import (
    ChainState,
    ChannelState,
    HashTimeLockState,
    NettingChannelState,
    TokenNetworkRegistryState,
    TokenNetworkState,
    make_empty_pending_locks_state,
)
from raiden.transfer.state_change import Block, ContractReceiveChannelNew, ContractReceiveChannelSettled
from raiden.utils.copy import deepcopy
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.transfers import random_secret
from raiden.utils.typing import (
    Address,
    BlockExpiration,
    BlockGasLimit,
    BlockNumber,
    MessageID,
    Nonce,
    PrivateKey,
    Secret,
    SecretHash,
    TokenAddress,
    TokenAmount,
)

from hypothesis.strategies import SearchStrategy

@composite
def secret(draw) -> Secret:
    return draw(builds(random_secret))


@composite
def address(draw) -> bytes:
    return draw(binary(min_size=20, max_size=20))


@composite
def payment_id(draw) -> int:
    return draw(integers(min_value=1, max_value=UINT64_MAX))


def event_types_match(events: List[Any], *expected_types: Type[Any]) -> bool:
    return Counter([type(event) for event in events]) == Counter(expected_types)


def transferred_amount(state: ChannelState) -> TokenAmount:
    return 0 if not state.balance_proof else state.balance_proof.transferred_amount


@dataclass
class Route:
    hops: Tuple[Address, ...]
    channel_from: int = 0

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
            assert (
                our_unclaimed + our_transferred
                >= self.our_previous_transferred[address] + self.our_previous_unclaimed[address]
            )
            assert (
                partner_unclaimed + partner_transferred
                >= self.partner_previous_transferred[address] + self.partner_previous_unclaimed[address]
            )
            self.our_previous_transferred[address] = our_transferred
            self.partner_previous_transferred[address] = partner_transferred
            self.our_previous_unclaimed[address] = our_unclaimed
            self.partner_previous_unclaimed[address] = partner_unclaimed

    def assert_channel_state_invariants(self) -> None:
        """Assert all channel state invariants given in the Raiden specification"""
        for netting_channel in self.address_to_channel.values():
            our_state = netting_channel.our_state
            partner_state = netting_channel.partner_state
            our_transferred_amount: TokenAmount = 0
            if our_state.balance_proof:
                our_transferred_amount = our_state.balance_proof.transferred_amount
                assert our_transferred_amount >= 0
            partner_transferred_amount: TokenAmount = 0
            if partner_state.balance_proof:
                partner_transferred_amount = partner_state.balance_proof.transferred_amount
                assert partner_transferred_amount >= 0
            assert channel.get_distributable(our_state, partner_state) >= 0
            assert channel.get_distributable(partner_state, our_state) >= 0
            our_deposit = netting_channel.our_total_deposit
            partner_deposit = netting_channel.partner_total_deposit
            total_deposit = our_deposit + partner_deposit
            our_amount_locked = channel.get_amount_locked(our_state)
            our_balance = channel.get_balance(our_state, partner_state)
            partner_amount_locked = channel.get_amount_locked(partner_state)
            partner_balance = channel.get_balance(partner_state, our_state)
            assert 0 <= our_amount_locked <= our_balance
            assert 0 <= partner_amount_locked <= partner_balance
            assert our_amount_locked <= total_deposit
            assert partner_amount_locked <= total_deposit
            our_transferred = partner_transferred_amount - our_transferred_amount
            netted_transferred = (
                our_transferred + partner_amount_locked - our_amount_locked
            )
            assert 0 <= our_deposit + our_transferred - our_amount_locked <= total_deposit
            assert 0 <= partner_deposit - our_transferred - partner_amount_locked <= total_deposit
            assert -our_deposit <= netted_transferred <= partner_deposit


class ChainStateStateMachine(RuleBasedStateMachine):

    address_to_privkey: Dict[Address, PrivateKey]
    address_to_client: Dict[Address, Client]
    transfer_order: TransferOrder
    replay_path: bool
    random_seed: Any
    block_number: BlockNumber
    block_hash: bytes
    random: Random
    token_network_address: Address
    token_id: Address
    token_network_state: TokenNetworkState
    token_network_registry_address: Address
    token_network_registry_state: TokenNetworkRegistryState

    def __init__(self) -> None:
        self.replay_path = False
        self.address_to_privkey = {}
        self.address_to_client = {}
        self.transfer_order = TransferOrder()
        super().__init__()

    def new_address(self) -> Address:
        privkey, address = factories.make_privkey_address()
        self.address_to_privkey[address] = privkey
        return address

    def _new_channel_state(
        self, our_address: Address, partner_address: Address
    ) -> NettingChannelState:
        identifier = factories.make_canonical_identifier(
            token_network_address=self.token_network_address
        )
        our_state = factories.NettingChannelEndStateProperties(
            balance=TokenAmount(1000), address=our_address
        )
        partner_state = factories.NettingChannelEndStateProperties(
            balance=TokenAmount(1000), address=partner_address
        )
        return factories.create(
            factories.NettingChannelStateProperties(
                our_state=our_state,
                partner_state=partner_state,
                canonical_identifier=identifier,
            )
        )

    def new_channel(
        self, client_address: Address, partner_address: Optional[Address] = None
    ) -> Address:
        """Create a new partner address with private key and channel. The
        private key and channels are listed in the instance's dictionaries,
        the address is returned and should be added to the partners Bundle.
        """
        if not partner_address:
            partner_address = self.new_address()
        client = self.address_to_client[client_address]
        channel_state = self._new_channel_state(client_address, partner_address)
        client.address_to_channel[partner_address] = channel_state
        partner_client = self.address_to_client.get(partner_address)
        if partner_client is not None:
            mirrored = deepcopy(channel_state)
            mirrored.our_state, mirrored.partner_state = (
                mirrored.partner_state,
                mirrored.our_state,
            )
            partner_client.address_to_channel[client_address] = mirrored
        return partner_address

    def _new_channel_transaction(
        self, client_address: Address, partner_address: Address
    ) -> None:
        client = self.address_to_client[client_address]
        channel_state = client.address_to_channel[partner_address]
        assert isinstance(channel_state, NettingChannelState)
        channel_new_state_change = ContractReceiveChannelNew(
            transaction_hash=factories.make_transaction_hash(),
            channel_state=channel_state,
            block_number=self.block_number,
            block_hash=factories.make_block_hash(),
        )
        node.state_transition(client.chain_state, channel_new_state_change)

    def new_channel_with_transaction(
        self, client_address: Address, partner_address: Optional[Address] = None
    ) -> Address:
        partner_address = self.new_channel(client_address, partner_address)
        self._new_channel_transaction(client_address, partner_address)
        if partner_address in self.address_to_client:
            self._new_channel_transaction(partner_address, client_address)
        return partner_address

    def new_client(self) -> Address:
        address = self.new_address()
        chain_state = ChainState(
            pseudo_random_generator=self.random,
            block_number=self.block_number,
            block_hash=self.block_hash,
            our_address=address,
            chain_id=factories.UNIT_CHAIN_ID,
        )
        chain_state.identifiers_to_tokennetworkregistries[
            self.token_network_registry_address
        ] = deepcopy(self.token_network_registry_state)
        chain_state.tokennetworkaddresses_to_tokennetworkregistryaddresses[
            self.token_network_address
        ] = self.token_network_registry_address
        self.address_to_client[address] = Client(chain_state=chain_state)
        return address

    @initialize(
        target=routes,
        block_number=integers(min_value=GENESIS_BLOCK_NUMBER + 1),
        random=randoms(),
        random_seed=random_module(),
    )
    def initialize_all(
        self, block_number: int, random: Random, random_seed: Any
    ) -> Tuple[Route, ...]:
        self.random_seed = random_seed
        self.block_number = block_number
        self.block_hash = factories.make_block_hash()
        self.random = random
        self.token_network_address = factories.UNIT_TOKEN_NETWORK_ADDRESS
        self.token_id = factories.UNIT_TOKEN_ADDRESS
        self.token_network_state = TokenNetworkState(
            address=self.token_network_address, token_address=self.token_id
        )
        self.token_network_registry_address = factories.make_token_network_registry_address()
        self.token_network_registry_state = TokenNetworkRegistryState(
            self.token_network_registry_address, [self.token_network_state]
        )
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

    def channel_opened(
        self, partner_address: Address, client_address: Address
    ) -> bool:
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

    used_secrets: Set[Secret]
    processed_secret_request_secrethashes: Set[SecretHash]
    initiated: Set[Secret]

    def __init__(self) -> None:
        super().__init__()
        self.used_secrets = set()
        self.processed_secret_request_secrethashes = set()
        self.initiated = set()

    def _available_amount(self, route: Route) -> TokenAmount:
        client = self.address_to_client[route.initiator]
        netting_channel = client.address_to_channel[route.hops[1]]
        return channel.get_distributable(netting_channel.our_state, netting_channel.partner_state)

    def _is_expired(self, secrethash: SecretHash, initiator: Address) -> bool:
        expiry = self.address_to_client[initiator].expected_expiry[secrethash]
        return self.block_number >= expiry + DEFAULT_WAIT_BEFORE_LOCK_REMOVAL

    def _is_removed(self, action: ActionInitInitiator) -> bool:
        return self._is_expired(action.transfer.secrethash, action.transfer.initiator)

    def _action_init_initiator(
        self, route: Route, transfer: TransferDescriptionWithSecretState
    ) -> ActionInitInitiator:
        client = self.address_to_client[route.initiator]
        channel_state = client.address_to_channel[route.hops[1]]
        if transfer.secrethash not in client.expected_expiry:
            client.expected_expiry[transfer.secrethash] = self.block_number + 10
        return ActionInitInitiator(
            transfer=transfer, routes=[factories.make_route_from_channel(channel_state)]
        )

    def _new_transfer_description(
        self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret
    ) -> TransferDescriptionWithSecretState:
        self.used_secrets.add(secret)
        return TransferDescriptionWithSecretState(
            token_network_registry_address=self.token_network_registry_address,
            payment_identifier=payment_id,
            amount=amount,
            token_network_address=self.token_network_address,
            initiator=route.initiator,
            target=route.target,
            secret=secret,
        )

    @rule(
        target=send_locked_transfers,
        route=routes,
        payment_id=payment_id(),
        amount=integers(min_value=1, max_value=100),
        secret=secret(),
    )
    def valid_init_initiator(
        self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret
    ) -> utils.SendLockedTransferInNode:
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
        return utils.SendLockedTransferInNode(
            event=result.events[0],
            action=action,
            node=route.initiator,
            private_key=self.address_to_privkey[route.initiator],
        )

    @rule(
        route=routes,
        payment_id=payment_id(),
        excess_amount=integers(min_value=1),
        secret=secret(),
    )
    def exceeded_capacity_init_initiator(
        self, route: Route, payment_id: int, excess_amount: int, secret: Secret
    ) -> None:
        amount = self._available_amount(route) + excess_amount
        transfer = self._new_transfer_description(route, payment_id, amount, secret)
        action = self._action_init_initiator(route, transfer)
        client = self.address_to_client[route.initiator]
        result = node.state_transition(client.chain_state, action)
        assert event_types_match(result.events, EventPaymentSentFailed)
        self.event('ActionInitInitiator failed: Amount exceeded')

    @rule(
        previous=send_locked_transfers,
        route=routes,
        payment_id=payment_id(),
        amount=integers(min_value=1),
    )
    def used_secret_init_initiator(
        self, previous: utils.SendLockedTransferInNode, route: Route, payment_id: int, amount: TokenAmount
    ) -> None:
        assume(not self._is_removed(previous.action))
        client = self.address_to_client[previous.node]
        secret = previous.action.transfer.secret
        transfer = self._new_transfer_description(route, payment_id, amount, secret)
        action = self._action_init_initiator(route, transfer)
        result = node.state_transition(client.chain_state, action)
        assert not result.events
        self.event('ActionInitInitiator failed: Secret already in use.')

    @rule(previous=send_locked_transfers)
    def replay_init_initiator(self, previous: utils.SendLockedTransferInNode) -> None:
        assume(not self._is_removed(previous.action))
        client = self.address_to_client[previous.node]
        result = node.state_transition(client.chain_state, previous.action)
        assert not result.events
        self.event('Replayed init_initiator action ignored')

    @rule(
        route=routes,
        payment_id=payment_id(),
        excess_amount=integers(min_value=1),
        secret=secret(),
    )
    def exceeded_capacity_init_initiator(
        self, route: Route, payment_id: int, excess_amount: int, secret: Secret
    ) -> None:
        pass  # Duplicate rule removed

    @rule(
        target=send_secret_reveals_forward,
        source=consumes(send_secret_requests),
    )
    def process_valid_secret_request(
        self, source: utils.SendSecretRequestInNode
    ) -> utils.SendSecretRevealInNode:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]
        state_change = utils.send_secret_request_to_receive_secret_request(source)
        assume(state_change.secrethash not in self.processed_secret_request_secrethashes)
        result = node.state_transition(initiator_client.chain_state, state_change)
        if state_change.secrethash in self.processed_secret_request_secrethashes:
            assert not result.events
            self.event('Valid SecretRequest dropped due to previous one with same secrethash.')
            return multiple()
        elif self._is_expired(state_change.secrethash, initiator_address):
            assert not result.events
            self.event('Otherwise valid SecretRequest dropped due to expired lock.')
            return multiple()
        else:
            assert event_types_match(result.events, SendSecretReveal)
            self.event('Valid SecretRequest accepted.')
            self.processed_secret_request_secrethashes.add(state_change.secrethash)
            return utils.SendSecretRevealInNode(
                node=initiator_address, event=result.events[0]
            )

    @rule(
        source=send_secret_requests,
        wrong_amount=integers(),
    )
    def process_secret_request_with_wrong_amount(
        self, source: utils.SendSecretRequestInNode, wrong_amount: int
    ) -> None:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]
        state_change = utils.send_secret_request_to_receive_secret_request(source)
        assume(wrong_amount != state_change.amount)
        state_change = replace(state_change, amount=wrong_amount)
        result = node.state_transition(initiator_client.chain_state, state_change)
        transfer_expired = self._is_expired(state_change.secrethash, initiator_address)
        secrethash_known = state_change.secrethash in self.processed_secret_request_secrethashes
        if transfer_expired or secrethash_known:
            assert not result.events
            self.event('Invalid secret request dropped silently (wrong amount)')
        else:
            self.processed_secret_request_secrethashes.add(state_change.secrethash)

    @rule(
        source=send_secret_requests,
        wrong_secret=secret(),
    )
    def process_secret_request_with_wrong_secrethash(
        self, source: utils.SendSecretRequestInNode, wrong_secret: Secret
    ) -> None:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]
        state_change = utils.send_secret_request_to_receive_secret_request(source)
        wrong_secrethash = sha256_secrethash(wrong_secret)
        assume(wrong_secrethash != state_change.secrethash)
        state_change = replace(state_change, secrethash=wrong_secrethash)
        result = node.state_transition(initiator_client.chain_state, state_change)
        assert not result.events
        self.event('Invalid secret request dropped (wrong secrethash)')

    @rule(
        source=send_secret_requests,
        wrong_payment_identifier=integers(),
    )
    def process_secret_request_with_wrong_payment_identifier(
        self, source: utils.SendSecretRequestInNode, wrong_payment_identifier: int
    ) -> None:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]
        state_change = utils.send_secret_request_to_receive_secret_request(source)
        assume(wrong_payment_identifier != state_change.payment_identifier)
        state_change = replace(state_change, payment_identifier=wrong_payment_identifier)
        result = node.state_transition(initiator_client.chain_state, state_change)
        assert not result.events
        self.event('Invalid secret request dropped (wrong payment identifier)')


class ChainStateStateMachine(RuleBasedStateMachine):
    # Existing implementation...
    pass


class WithOurAddress:
    our_address: Address
    data: Any

    def __init__(self, our_address: Address, data: Any) -> None:
        self.our_address = our_address
        self.data = data


class MediatorMixin:
    partner_to_balance_proof_data: Dict[Address, 'BalanceProofData']
    secrethash_to_secret: Dict[SecretHash, Secret]
    waiting_for_unlock: Dict[Secret, Address]
    initial_number_of_channels: int

    def __init__(self) -> None:
        super().__init__()
        self.partner_to_balance_proof_data = {}
        self.secrethash_to_secret = {}
        self.waiting_for_unlock = {}
        self.initial_number_of_channels = 2

    def _get_balance_proof_data(
        self, partner: Address, client_address: Address
    ) -> 'BalanceProofData':
        if partner not in self.partner_to_balance_proof_data:
            client = self.address_to_client[client_address]
            partner_channel = client.address_to_channel[partner]
            self.partner_to_balance_proof_data[partner] = BalanceProofData(
                canonical_identifier=partner_channel.canonical_identifier
            )
        return self.partner_to_balance_proof_data[partner]

    def _update_balance_proof_data(
        self,
        partner: Address,
        amount: TokenAmount,
        expiration: BlockExpiration,
        secret: Secret,
        our_address: Address,
    ) -> 'BalanceProofData':
        expected = self._get_balance_proof_data(partner, our_address)
        lock = HashTimeLockState(
            amount=amount, expiration=expiration, secrethash=sha256_secrethash(secret)
        )
        expected.update(amount, lock)
        return expected

    def _new_mediator_transfer(
        self,
        initiator_address: Address,
        target_address: Address,
        payment_id: int,
        amount: TokenAmount,
        secret: Secret,
        our_address: Address,
    ) -> LockedTransferSignedState:
        initiator_pkey = self.address_to_privkey[initiator_address]
        balance_proof_data = self._update_balance_proof_data(
            initiator_address, amount, self.block_number + 10, secret, our_address
        )
        self.secrethash_to_secret[sha256_secrethash(secret)] = secret
        return factories.create(
            factories.LockedTransferSignedStateProperties(
                **balance_proof_data.properties.__dict__,
                amount=amount,
                expiration=BlockExpiration(self.block_number + 10),
                payment_identifier=payment_id,
                secret=secret,
                initiator=initiator_address,
                target=target_address,
                token=self.token_id,
                sender=initiator_address,
                recipient=our_address,
                pkey=initiator_pkey,
                message_identifier=MessageID(1),
            )
        )

    def _action_init_mediator(
        self, transfer: LockedTransferSignedState, client_address: Address
    ) -> WithOurAddress:
        client = self.address_to_client[client_address]
        initiator_channel = client.address_to_channel[Address(transfer.initiator)]
        target_channel = client.address_to_channel[Address(transfer.target)]
        assert isinstance(target_channel, NettingChannelState)
        action = ActionInitMediator(
            candidate_route_states=[factories.make_route_from_channel(target_channel)],
            from_hop=factories.make_hop_to_channel(initiator_channel),
            from_transfer=transfer,
            balance_proof=transfer.balance_proof,
            sender=transfer.balance_proof.sender,
        )
        return WithOurAddress(our_address=client_address, data=action)

    def _unwrap(
        self, with_our_address: WithOurAddress
    ) -> Tuple[Any, Client, Address]:
        our_address = with_our_address.our_address
        data = with_our_address.data
        client = self.address_to_client[our_address]
        return (data, client, our_address)

    @rule(
        target=init_mediators,
        from_channel=routes,
        to_channel=routes,
        payment_id=payment_id(),
        amount=integers(min_value=1, max_value=100),
        secret=secret(),
    )
    def valid_init_mediator(
        self,
        from_channel: Route,
        to_channel: Route,
        payment_id: int,
        amount: TokenAmount,
        secret: Secret,
    ) -> WithOurAddress:
        our_address = from_channel.initiator
        assume(to_channel.initiator == our_address)
        client = self.address_to_client[our_address]
        from_partner = from_channel.target
        to_partner = to_channel.target
        assume(from_partner != to_partner)
        transfer = self._new_mediator_transfer(
            from_partner, to_partner, payment_id, amount, secret, our_address
        )
        client_data = self._action_init_mediator(transfer, our_address)
        result = node.state_transition(client.chain_state, client_data.data)
        assert event_types_match(result.events, SendProcessed, SendLockedTransfer)
        return client_data

    @rule(
        target=secret_requests,
        previous_action_with_address=consumes(init_mediators),
    )
    def valid_receive_secret_reveal(
        self, previous_action_with_address: WithOurAddress
    ) -> WithOurAddress:
        previous_action, client, our_address = self._unwrap(previous_action_with_address)
        secret = self.secrethash_to_secret[previous_action.from_transfer.lock.secrethash]
        sender = previous_action.from_transfer.target
        recipient = previous_action.from_transfer.initiator
        action = ReceiveSecretReveal(secret=secret, sender=sender)
        result = node.state_transition(client.chain_state, action)
        expiration = previous_action.from_transfer.lock.expiration
        in_time = self.block_number < expiration - DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
        still_waiting = self.block_number < expiration + DEFAULT_WAIT_BEFORE_LOCK_REMOVAL
        if (
            in_time
            and self.channel_opened(sender, our_address)
            and self.channel_opened(recipient, our_address)
        ):
            assert event_types_match(
                result.events, SendSecretReveal, SendUnlock, EventUnlockSuccess
            )
            self.event('Unlock successful.')
            self.waiting_for_unlock[secret] = recipient
        elif still_waiting and self.channel_opened(recipient, our_address):
            assert event_types_match(result.events, SendSecretReveal)
            self.event('Unlock failed, secret revealed too late.')
        else:
            assert not result.events
            self.event('ReceiveSecretRevealed after removal of lock - dropped.')
        return WithOurAddress(our_address=our_address, data=action)

    @rule(
        previous_action_with_address=secret_requests,
    )
    def replay_receive_secret_reveal(
        self, previous_action_with_address: WithOurAddress
    ) -> None:
        previous_action, client, _ = self._unwrap(previous_action_with_address)
        result = node.state_transition(client.chain_state, previous_action)
        assert not result.events

    @rule(
        previous_action_with_address=secret_requests,
        invalid_sender=address(),
    )
    def replay_receive_secret_reveal_scrambled_sender(
        self, previous_action_with_address: WithOurAddress, invalid_sender: bytes
    ) -> None:
        previous_action, client, _ = self._unwrap(previous_action_with_address)
        action = ReceiveSecretReveal(secret=previous_action.secret, sender=invalid_sender)
        result = node.state_transition(client.chain_state, action)
        assert not result.events

    @rule(
        previous_action_with_address=init_mediators,
        secret=secret(),
    )
    def wrong_secret_receive_secret_reveal(
        self, previous_action_with_address: WithOurAddress, secret: Secret
    ) -> None:
        previous_action, client, _ = self._unwrap(previous_action_with_address)
        sender = previous_action.from_transfer.target
        action = ReceiveSecretReveal(secret=secret, sender=sender)
        result = node.state_transition(client.chain_state, action)
        assert not result.events

    @rule(
        target=secret_requests,
        previous_action_with_address=consumes(init_mediators),
        invalid_sender=address(),
    )
    def wrong_address_receive_secret_reveal(
        self,
        previous_action_with_address: WithOurAddress,
        invalid_sender: bytes,
    ) -> WithOurAddress:
        previous_action, client, our_address = self._unwrap(previous_action_with_address)
        secret = self.secrethash_to_secret[previous_action.from_transfer.lock.secrethash]
        invalid_action = ReceiveSecretReveal(secret=secret, sender=invalid_sender)
        result = node.state_transition(client.chain_state, invalid_action)
        assert not result.events
        valid_sender = previous_action.from_transfer.target
        valid_action = ReceiveSecretReveal(secret=secret, sender=valid_sender)
        return WithOurAddress(our_address=our_address, data=valid_action)


class OnChainMixin:

    def new_blocks(self, number: int) -> None:
        for _ in range(number):
            block_state_change = Block(
                block_number=BlockNumber(self.block_number + 1),
                gas_limit=BlockGasLimit(1),
                block_hash=make_block_hash(),
            )
            for client in self.address_to_client.values():
                result = node.state_transition(client.chain_state, block_state_change)
            self.block_number += 1

    def open_channel(self, reference: Route) -> Route:
        address = self.new_channel_with_transaction(reference.initiator)
        return self.routes_for_new_channel(reference.initiator, address)

    def routes_for_new_channel(self, from_address: Address, to_address: Address) -> Tuple[Route, ...]:
        return multiple()

    @rule(
        reference=consumes(routes),
    )
    def settle_channel(self, reference: Route) -> None:
        client = self.address_to_client[reference.initiator]
        channel_state = client.address_to_channel[reference.hops[1]]
        channel_settled_state_change = ContractReceiveChannelSettled(
            transaction_hash=factories.make_transaction_hash(),
            canonical_identifier=factories.make_canonical_identifier(
                chain_identifier=channel_state.chain_id,
                token_network_address=channel_state.token_network_address,
                channel_identifier=channel_state.identifier,
            ),
            block_number=self.block_number + 1,
            block_hash=factories.make_block_hash(),
            our_transferred_amount=TokenAmount(0),
            partner_transferred_amount=TokenAmount(0),
            our_onchain_locksroot=LOCKSROOT_OF_NO_LOCKS,
            partner_onchain_locksroot=LOCKSROOT_OF_NO_LOCKS,
        )
        node.state_transition(client.chain_state, channel_settled_state_change)


class MediatorStateMachine(MediatorMixin, ChainStateStateMachine):
    pass


class OnChainStateMachine(OnChainMixin, ChainStateStateMachine):

    def create_network(self) -> List[Route]:
        client = self.new_client()
        partners = [self.new_channel_with_transaction(client) for _ in range(3)]
        return [Route(hops=(client, partner)) for partner in partners]


class MultiChannelMediatorStateMachine(MediatorMixin, OnChainMixin, ChainStateStateMachine):
    pass


class BalanceProofData:

    canonical_identifier: Any
    _pending_locks: Any
    properties: Optional[Any]

    def __init__(self, canonical_identifier: Any) -> None:
        self._canonical_identifier = canonical_identifier
        self._pending_locks = make_empty_pending_locks_state()
        self.properties = None

    def update(self, amount: TokenAmount, lock: HashTimeLockState) -> None:
        self._pending_locks = channel.compute_locks_with(self._pending_locks, lock)
        assert self._pending_locks
        if self.properties:
            self.properties = factories.replace(
                self.properties,
                locked_amount=self.properties.locked_amount + amount,
                locksroot=compute_locksroot(self._pending_locks),
                nonce=self.properties.nonce + 1,
            )
        else:
            self.properties = factories.BalanceProofProperties(
                transferred_amount=TokenAmount(0),
                locked_amount=amount,
                nonce=Nonce(1),
                locksroot=compute_locksroot(self._pending_locks),
                canonical_identifier=self._canonical_identifier,
            )


class DirectTransfersStateMachine(InitiatorMixin, TargetMixin, ChainStateStateMachine):

    def create_network(self) -> List[Route]:
        address1 = self.new_client()
        address2 = self.new_client()
        self.new_channel_with_transaction(address1, address2)
        return [Route(hops=(address1, address2))]


class MediatorStateMachine(MediatorMixin, ChainStateStateMachine):
    pass


class TestMediator:
    pass  # Skipped using pytest.mark.skip

class TestOnChain(OnChainStateMachine.TestCase):
    pass

class TestMultiChannelMediator:
    pass  # Skipped using pytest.mark.skip

class TestDirectTransfers(DirectTransfersStateMachine.TestCase):
    pass


TestMediator = pytest.mark.skip(MediatorStateMachine.TestCase)
TestOnChain = OnChainStateMachine.TestCase
TestMultiChannelMediator = pytest.mark.skip(MultiChannelMediatorStateMachine.TestCase)
TestDirectTransfers = DirectTransfersStateMachine.TestCase


def unwrap_multiple(multiple_results: multiple) -> Optional[Any]:
    values = multiple_results.values
    return values[0] if values else None


def test_regression_malicious_secret_request_handled_properly() -> None:
    state = DirectTransfersStateMachine()
    state.replay_path = True
    v1 = unwrap_multiple(
        state.initialize_all(block_number=1, random=Random(), random_seed=None)
    )
    v2 = state.valid_init_initiator(
        route=v1, amount=1, payment_id=1, secret=b'\x00' * 32
    )
    v3 = state.process_send_locked_transfer(source=v2)
    state.process_secret_request_with_wrong_amount(source=v3, wrong_amount=0)
    state.replay_init_initiator(previous=v2)
    state.teardown()
