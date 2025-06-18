from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Dict, List, Set, Optional, Tuple, Union

import pytest
from hypothesis import assume, event
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    consumes,
    initialize,
    invariant,
    multiple,
    rule,
)
from hypothesis.strategies import binary, builds, composite, integers, random_module, randoms

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
from raiden.transfer.state_change import (
    Block,
    ContractReceiveChannelNew,
    ContractReceiveChannelSettled,
)
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


@composite
def secret(draw) -> Secret:
    return draw(builds(random_secret))


@composite
def address(draw) -> Address:
    return draw(binary(min_size=20, max_size=20))


@composite
def payment_id(draw) -> int:
    return draw(integers(min_value=1, max_value=UINT64_MAX))


def event_types_match(events: List[Any], *expected_types: Any) -> bool:
    return Counter([type(event) for event in events]) == Counter(expected_types)


def transferred_amount(state: Any) -> TokenAmount:
    return TokenAmount(0) if not state.balance_proof else state.balance_proof.transferred_amount


@dataclass
class Route:
    hops: List[Address]
    channel_from: int = 0

    @property
    def initiator(self) -> Address:
        return self.hops[0]

    @property
    def target(self) -> Address:
        return self.hops[-1]


# shared bundles of ChainStateStateMachine and all mixin classes
routes = Bundle("routes")
init_initiators = Bundle("init_initiators")
init_mediators = Bundle("init_mediators")
send_locked_transfers = Bundle("send_locked_transfers")
secret_requests = Bundle("secret_requests")
send_secret_requests = Bundle("send_secret_requests")
send_secret_reveals_backward = Bundle("send_secret_reveals_backward")
send_secret_reveals_forward = Bundle("send_secret_reveals_forward")
send_unlocks = Bundle("send_unlocks")

AddressToAmount = Dict[Address, TokenAmount]


def make_tokenamount_defaultdict() -> defaultdict:
    return defaultdict(lambda: TokenAmount(0))


@dataclass
class TransferOrder:
    initiated: List[SecretHash] = field(default_factory=list)
    answered: List[SecretHash] = field(default_factory=list)
    # TODO generalize this to channels with multiple routes


@dataclass
class Client:
    chain_state: ChainState

    address_to_channel: Dict[Address, NettingChannelState] = field(default_factory=dict)
    expected_expiry: Dict[SecretHash, BlockNumber] = field(default_factory=dict)
    our_previous_deposit: AddressToAmount = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_deposit: AddressToAmount = field(default_factory=make_tokenamount_defaultdict)
    our_previous_transferred: AddressToAmount = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_transferred: AddressToAmount = field(
        default_factory=make_tokenamount_defaultdict
    )
    our_previous_unclaimed: AddressToAmount = field(default_factory=make_tokenamount_defaultdict)
    partner_previous_unclaimed: AddressToAmount = field(
        default_factory=make_tokenamount_defaultdict
    )
    transfer_order: TransferOrder = field(default_factory=TransferOrder)

    def assert_monotonicity_invariants(self) -> None:
        """Assert all monotonicity properties stated in Raiden specification"""
        for (
            address,
            netting_channel,
        ) in self.address_to_channel.items():  # pylint: disable=no-member

            # constraint (1TN)
            assert netting_channel.our_total_deposit >= self.our_previous_deposit[address]
            assert netting_channel.partner_total_deposit >= self.partner_previous_deposit[address]
            self.our_previous_deposit[address] = netting_channel.our_total_deposit
            self.partner_previous_deposit[address] = netting_channel.partner_total_deposit

            # TODO add constraint (2TN) when withdrawal is implemented
            # constraint (3R) and (4R)
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
                >= self.partner_previous_transferred[address]
                + self.partner_previous_unclaimed[address]
            )
            self.our_previous_transferred[address] = our_transferred
            self.partner_previous_transferred[address] = partner_transferred
            self.our_previous_unclaimed[address] = our_unclaimed
            self.partner_previous_unclaimed[address] = partner_unclaimed

    def assert_channel_state_invariants(self) -> None:
        """Assert all channel state invariants given in the Raiden specification"""
        for netting_channel in self.address_to_channel.values():  # pylint: disable=no-member
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

            # invariant (5.1R), add withdrawn amounts when implemented
            assert TokenAmount(0) <= our_amount_locked <= our_balance
            assert TokenAmount(0) <= partner_amount_locked <= partner_balance
            assert our_amount_locked <= total_deposit
            assert partner_amount_locked <= total_deposit

            our_transferred = partner_transferred_amount - our_transferred_amount
            netted_transferred = our_transferred + partner_amount_locked - our_amount_locked

            # invariant (6R), add withdrawn amounts when implemented
            assert TokenAmount(0) <= our_deposit + our_transferred - our_amount_locked <= total_deposit
            assert TokenAmount(0) <= partner_deposit - our_transferred - partner_amount_locked <= total_deposit

            # invariant (7R), add withdrawn amounts when implemented
            assert -our_deposit <= netted_transferred <= partner_deposit


class ChainStateStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        self.replay_path: bool = False
        self.address_to_privkey: Dict[Address, PrivateKey] = {}
        self.address_to_client: Dict[Address, Client] = {}
        self.transfer_order = TransferOrder()
        super().__init__()

    def new_address(self) -> Address:
        privkey, address = factories.make_privkey_address()
        self.address_to_privkey[address] = privkey
        return address

    def _new_channel_state(self, our_address: Address, partner_address: Address) -> NettingChannelState:
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
                our_state=our_state, partner_state=partner_state, canonical_identifier=identifier
            )
        )

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
            mirrored.our_state, mirrored.partner_state = mirrored.partner_state, mirrored.our_state
            partner_client.address_to_channel[client_address] = mirrored

        return partner_address

    def _new_channel_transaction(self, client_address: Address, partner_address: Address) -> None:
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
    def initialize_all(self, block_number: BlockNumber, random: Random, random_seed: Any) -> multiple:
        self.random_seed = random_seed

        self.block_number = block_number
        self.block_hash = factories.make_block_hash()
        self.random = random

        self.token_network_address = factories.UNIT_TOKEN_NETWORK_ADDRESS
        self.token_id = factories.UNIT_TOKEN_ADDRESS
        self.token_network_state = TokenNetworkState(
            address=self.token_network_address,
            token_address=self.token_id,
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

    def channel_opened(self, partner_address: Address, client_address: Address) -> bool:
        try:
            client = self.address_to_client[client_address]
        except KeyError:
            return False
        else:
            needed_channel = client.address_to_channel[partner_address]
            return channel.get_status(needed_channel) == ChannelState.STATE_OPENED

    def create_network(self) -> List[Route]:
        raise NotImplementedError("Every fuzz test needs to override this.")


class InitiatorMixin:
    address_to_client: Dict[Address, Client]
    block_number: BlockNumber

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

    def _is_removed(self, action: Any) -> bool:
        return self._is_expired(action.transfer.secrethash, action.transfer.initiator)

    def _action_init_initiator(self, route: Route, transfer: TransferDescriptionWithSecretState) -> ActionInitInitiator:
        client = self.address_to_client[route.initiator]
        channel = client.address_to_channel[route.hops[1]]
        if transfer.secrethash not in client.expected_expiry:
            client.expected_expiry[transfer.secrethash] = self.block_number + 10
        return ActionInitInitiator(transfer, [factories.make_route_from_channel(channel)])

    def _new_transfer_description(self, route: Route, payment_id: int, amount: TokenAmount, secret: Secret) -> TransferDescriptionWithSecretState:
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
        payment_id=payment_id(),  # pylint: disable=no-value-for-parameter
        amount=integers(min_value=1, max_value=100),
        secret=secret(),  # pylint: disable=no-value-for-parameter
    )
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

        return utils.SendLockedTransferInNode(
            event=result.events[0],
            action=action,
            node=route.initiator,
            private_key=self.address_to_privkey[route.initiator],
        )

    @rule(
        route=routes,
        payment_id=payment_id(),  # pylint: disable=no-value-for-parameter
        excess_amount=integers(min_value=1),
        secret=secret(),  # pylint: disable=no-value-for-parameter
    )
    def exceeded_capacity_init_initiator(self, route: Route, payment_id: int, excess_amount: TokenAmount, secret: Secret) -> None:
        amount = self._available_amount(route) + excess_amount
        transfer = self._new_transfer_description(route, payment_id, amount, secret)
        action = self._action_init_initiator(route, transfer)
        client = self.address_to_client[route.initiator]
        result = node.state_transition(client.chain_state, action)
        assert event_types_match(result.events, EventPaymentSentFailed)
        self.event("ActionInitInitiator failed: Amount exceeded")

    @rule(
        previous=send_locked_transfers,
        route=routes,
        payment_id=payment_id(),  # pylint: disable=no-value-for-parameter
        amount=integers(min_value=1),
    )
    def used_secret_init_initiator(self, previous: utils.SendLockedTransferInNode, route: Route, payment_id: int, amount: TokenAmount) -> None:
        assume(not self._is_removed(previous.action))

        client = self.address_to_client[previous.node]
        secret = previous.action.transfer.secret

        transfer = self._new_transfer_description(route, payment_id, amount, secret)
        action = self._action_init_initiator(route, transfer)
        result = node.state_transition(client.chain_state, action)

        assert not result.events
        self.event("ActionInitInitiator failed: Secret already in use.")

    @rule(previous=send_locked_transfers)
    def replay_init_initiator(self, previous: utils.SendLockedTransferInNode) -> None:
        assume(not self._is_removed(previous.action))

        client = self.address_to_client[previous.node]
        result = node.state_transition(client.chain_state, previous.action)

        assert not result.events
        self.event("Replayed init_initiator action ignored")

    @rule(target=send_secret_reveals_forward, source=consumes(send_secret_requests))
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
            self.event("Valid SecretRequest dropped due to previous one with same secrethash.")
            return multiple()
        elif self._is_expired(state_change.secrethash, initiator_address):
            assert not result.events
            self.event("Otherwise valid SecretRequest dropped due to expired lock.")
            return multiple()
        else:
            assert event_types_match(result.events, SendSecretReveal)
            self.event("Valid SecretRequest accepted.")
            self.processed_secret_request_secrethashes.add(state_change.secrethash)
            return utils.SendSecretRevealInNode(node=initiator_address, event=result.events[0])

    @rule(source=send_secret_requests, wrong_amount=integers())
    def process_secret_request_with_wrong_amount(
        self, source: utils.SendSecretRequestInNode, wrong_amount: TokenAmount
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
            self.event("Invalid secret request dropped silently (wrong amount)")
        else:
            self.processed_secret_request_secrethashes.add(state_change.secrethash)

    @rule(source=send_secret_requests, wrong_secret=secret())
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
        self.event("Invalid secret request dropped (wrong secrethash)")

    @rule(source=send_secret_requests, wrong_payment_identifier=integers())
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
        self.event("Invalid secret request dropped (wrong payment identifier)")

    @rule(target=send_unlocks, source=consumes(send_secret_reveals_backward))
    def process_secret_reveal_as_initiator(
        self, source: utils.SendSecretRevealInNode
    ) -> utils.SendUnlockInNode:
        initiator_address = source.event.recipient
        private_key = self.address_to_privkey[initiator_address]
        initiator_client = self.address_to_client[initiator_address]

        state_change = utils.send_secret_reveal_to_recieve_secret_reveal(source)
        result = node.state_transition(initiator_client.chain_state, state_change)

        assert event_types_match(
            result.events, SendUnlock, EventPaymentSentSuccess, EventUnlockSuccess
        )
        self.event("Valid secret reveal processed in initiator node.")

        return utils.SendUnlockInNode(
            node=initiator_address, private_key=private_key, event=result.events[0]
        )

    @rule(source=send_secret_reveals_backward, wrong_secret=secret())
    def process_secret_reveal_with_mismatched_secret_as_initiator(
        self, source: utils.SendSecretRevealInNode, wrong_secret: Secret
    ) -> None:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]

        state_change = utils.send_secret_reveal_to_recieve_secret_reveal(source)
        assume(state_change.secret != wrong_secret)
        state_change = replace(state_change, secret=wrong_secret)
        result = node.state_transition(initiator_client.chain_state, state_change)

        assert not result.events
        self.event("Secret reveal with wrong secret dropped in initiator node.")

    @rule(source=send_secret_reveals_backward, wrong_secret=secret())
    def process_secret_reveal_with_unknown_secrethash_as_initiator(
        self, source: utils.SendSecretRevealInNode, wrong_secret: Secret
    ) -> None:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]

        state_change = utils.send_secret_reveal_to_recieve_secret_reveal(source)
        assume(state_change.secret != wrong_secret)
        wrong_secrethash = sha256_secrethash(wrong_secret)
        state_change = replace(state_change, secret=wrong_secret, secrethash=wrong_secrethash)
        result = node.state_transition(initiator_client.chain_state, state_change)

        assert not result.events
        self.event("Secret reveal with unknown secrethash dropped in initiator node.")

    @rule(source=send_secret_reveals_backward, wrong_channel_id=integers())
    def process_secret_reveal_with_wrong_channel_identifier_as_initiator(
        self, source: utils.SendSecretRevealInNode, wrong_channel_id: int
    ) -> None:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]

        state_change = utils.send_secret_reveal_to_recieve_secret_reveal(source)
        assume(state_change.canonical_id.channel_id != wrong_channel_id)
        wrong_canonical_id = replace(state_change.canonical_id, channel_id=wrong_channel_id)
        state_change = replace(state_change, canonical_id=wrong_canonical_id)
        result = node.state_transition(initiator_client.chain_state, state_change)

        assert not result.events
        self.event("Secret reveal with unknown channel id dropped in initiator node.")

    @rule(
        source=send_secret_reveals_backward, wrong_channel_id=integers(), wrong_recipient=address()
    )
    def process_secret_reveal_with_wrong_queue_identifier_as_initiator(
        self, source: utils.SendSecretRevealInNode, wrong_channel_id: int, wrong_recipient: Address
    ) -> None:
        initiator_address = source.event.recipient
        initiator_client = self.address_to_client[initiator_address]

        state_change = utils.send_secret_reveal_to_recieve_secret_reveal(source)
        assume(state_change.canonical_id.channel_id != wrong_channel_id)
        wrong_canonical_id = replace(
            state_change.queue_id.canonical_id, channel_id=wrong_channel_id
        )
        wrong_queue_id = replace(
            state_change.queue_id, canonical_id=wrong_canonical_id, recipient=wrong_recipient
        )
        state_change = replace(state_change, queue_id=wrong_queue_id)
        result = node.state_transition(initiator_client.chain_state, state_change)

        assert not result.events
        self.event("Secret reveal with unknown queue id dropped in initiator node.")


class TargetMixin:
    @rule(target=send_secret_requests, source=consumes(send_locked_transfers))
    def process_send_locked_transfer(
        self, source: utils.SendLockedTransferInNode
    ) -> utils.SendSecretRequestInNode:
        target_address = source.event.recipient
        target_client = self.address_to_client[target_address]

        if not self.replay_path:
            assume(source.action.transfer.secrethash == self.transfer_order.initiated[0])

        message = utils.send_lockedtransfer_to_locked_transfer(source)
        action = utils.locked_transfer_to_action_init_target(message)

        result = node.state_transition(target_client.chain_state, action)

        assert event_types_match(result.events, SendProcessed, SendSecretRequest)

        self.transfer_order.answered.append(self.transfer_order.initiated.pop(0))

        return utils.SendSecretRequestInNode(result.events[1], target_address)

    @rule(source=send_locked_transfers, scrambling=utils.balance_proof_scrambling())
    def process_send_locked_transfer_with_scrambled_balance_proof(
        self, source: utils.SendLockedTransferInNode, scrambling: Dict[str, Any]
    ) -> None:
        target_address = source.event.recipient
        target_client = self.address_to_client[target_address]

        scrambled_balance_proof = replace(source.event.balance_proof, **scrambling.kwargs)
        assume(scrambled_balance_proof != source.event.balance_proof)
        scrambled_transfer = replace(source.event.transfer, balance_proof=scrambled_balance_proof)
        scrambled_event = replace(source.event, transfer=scrambled_transfer)
        scrambled_source = replace(source, event=scrambled_event)

        message = utils.send_lockedtransfer_to_locked_transfer(scrambled_source)
        action = utils.locked_transfer_to_action_init_target(message)
        result = node.state_transition(target_client.chain_state, action)

        if scrambling.field == "canonical_identifier":
            assert not result.events
            self.event("SendLockedTransfer with wrong channel identifier dropped in target node.")
        else:
            assert event_types_match(
                result.events, EventInvalidReceivedLockedTransfer, EventUnlockClaimFailed
            )
            self.event("SendLockedTransfer with scrambled balance proof caught in target node.")

    @rule(source=send_locked_transfers, scrambling=utils.hash_time_lock_scrambling())
    def process_send_locked_transfer_with_scrambled_hash_time_lock_state(
        self, source: utils.SendLockedTransferInNode, scrambling: utils.Scrambling
    ) -> None:
        target_address = source.event.recipient
        target_client = self.address_to_client[target_address]

        scrambled_lock = replace(source.event.transfer.lock, **scrambling.kwargs)
        assume(scrambled_lock != source.event.transfer.lock)
        scrambled_transfer = replace(source.event.transfer, lock=scrambled_lock)
        scrambled_event = replace(source.event, transfer=scrambled_transfer)
        scrambled_source = replace(source, event=scrambled_event)

        message = utils.send_lockedtransfer_to_locked_transfer(scrambled_source)
        action = utils.locked_transfer_to_action_init_target(message)
        result = node.state_transition(target_client.chain_state, action)

        assert event_types_match(
            result.events, EventInvalidReceivedLockedTransfer, EventUnlockClaimFailed
        )
        self.event("SendLockedTransfer with scrambled lock caught in target node.")

    @rule(source=send_locked_transfers, scrambling=utils.locked_transfer_scrambling())
    def process_send_locked_transfer_with_scrambled_locked_transfer_parameter(
        self, source: utils.SendLockedTransferInNode, scrambling: utils.Scrambling
    ) -> None:
        target_address = source.event.recipient
        target_client = self.address_to_client[target_address]

        message = utils.send_lockedtransfer_to_locked_transfer(source)
        scrambled_message = replace(message, **scrambling.kwargs)
        assume(scrambled_message != message)
        action = utils.locked_transfer_to_action_init_target(scrambled_message)
        result = node.state_transition(target_client.chain_state, action)

        if scrambling.field in ("token_network_address", "channel_identifier"):
            assert not result.events
            self.event("SendLockedTransfer with token network or channel dropped.")
        else:
            assert event_types_match(
                result.events, EventInvalidReceivedLockedTransfer, EventUnlockClaimFailed
            )
            self.event("SendLockedTransfer with scrambled parameter caught in target node.")

    @rule(target=send_secret_reveals_backward, source=consumes(send_secret_reveals_forward))
    def process_secret_reveal_as_target(
        self, source: utils.SendSecretRevealInNode
    ) -> utils.SendSecretRevealInNode:
        state_change = utils.send_secret_reveal_to_recieve_secret_reveal(source)
        target_address = source.event.recipient
        target_client = self.address_to_client[target_address]

        result = node.state_transition(target_client.chain_state, state_change)
        assert event_types_match(result.events, SendSecretReveal)
        self.event("Valid SecretReveal processed in target node.")

        return utils.SendSecretRevealInNode(node=target_address, event=result.events[0])

    @rule(source=send_secret_reveals_forward, wrong_secret=secret())
    def process_secret_reveal_with_mismatched_secret_as_target(
        self, source: utils.SendSecretRevealInNode, wrong_secret: Secret
    ) -> None:
        state_change = utils.send_secret_reveal_to_recieve_secret_reveal(source)
        assume(state_change.secret != wrong_secret)
        state_change = replace(state_change, secret=wrong_secret)

        target_address = source.event.recipient
        target_client = self.address_to_client[target_address]
        result = node.state_transition(target_client.chain_state, state_change)
        assert not result.events
        self.event("SecretReveal with wrong secret dropped in target node.")

    @rule(source