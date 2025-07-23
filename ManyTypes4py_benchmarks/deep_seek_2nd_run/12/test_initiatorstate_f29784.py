import random
import uuid
from typing import NamedTuple, Dict, List, Optional, Tuple, Any, Set, Union
from unittest.mock import patch
import pytest
from eth_utils import keccak
from raiden.constants import EMPTY_HASH, MAXIMUM_PENDING_TRANSFERS
from raiden.raiden_service import initiator_init
from raiden.settings import DEFAULT_MEDIATION_FEE_MARGIN, MAX_MEDIATION_FEE_PERC
from raiden.tests.utils import factories
from raiden.tests.utils.events import search_for_item
from raiden.tests.utils.factories import EMPTY, UNIT_SECRET, UNIT_TOKEN_NETWORK_ADDRESS, UNIT_TRANSFER_AMOUNT, UNIT_TRANSFER_IDENTIFIER, UNIT_TRANSFER_INITIATOR, UNIT_TRANSFER_TARGET, ChannelSet, TransferDescriptionProperties, create
from raiden.tests.utils.mocks import MockRaidenService
from raiden.tests.utils.transfer import assert_dropped
from raiden.transfer import channel
from raiden.transfer.architecture import State
from raiden.transfer.events import EventInvalidReceivedLockExpired, EventInvalidSecretRequest, EventPaymentSentFailed, EventPaymentSentSuccess, SendProcessed
from raiden.transfer.mediated_transfer import initiator, initiator_manager
from raiden.transfer.mediated_transfer.events import EventRouteFailed, EventUnlockFailed, EventUnlockSuccess, SendLockedTransfer, SendLockExpired, SendSecretReveal, SendUnlock
from raiden.transfer.mediated_transfer.initiator import calculate_fee_margin, calculate_safe_amount_with_fee
from raiden.transfer.mediated_transfer.state import InitiatorPaymentState, InitiatorTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ActionTransferReroute, ReceiveLockExpired, ReceiveSecretRequest, ReceiveSecretReveal, ReceiveTransferCancelRoute
from raiden.transfer.state import HashTimeLockState, NettingChannelState, RouteState, message_identifier_from_prng
from raiden.transfer.state_change import ActionCancelPayment, Block, ContractReceiveChannelClosed, ContractReceiveSecretReveal
from raiden.utils import typing
from raiden.utils.copy import deepcopy
from raiden.utils.transfers import random_secret
from raiden.utils.typing import Address, BlockNumber, FeeAmount, PaymentAmount, TokenAmount

def get_transfer_at_index(payment_state: InitiatorPaymentState, index: int) -> InitiatorTransferState:
    key = list(payment_state.initiator_transfers.keys())[index]
    return payment_state.initiator_transfers[key]

def make_initiator_manager_state(
    channels: ChannelSet,
    pseudo_random_generator: random.Random,
    transfer_description: Optional[TransferDescriptionProperties] = None,
    block_number: BlockNumber = BlockNumber(1),
    estimated_fee: FeeAmount = FeeAmount(0)
) -> InitiatorPaymentState:
    init = ActionInitInitiator(
        transfer=transfer_description or factories.UNIT_TRANSFER_DESCRIPTION,
        routes=channels.get_routes(estimated_fee=estimated_fee)
    )
    initial_state = None
    iteration = initiator_manager.state_transition(
        payment_state=initial_state,
        state_change=init,
        channelidentifiers_to_channels=channels.channel_map,
        addresses_to_channel=channels.addresses_to_channel(),
        pseudo_random_generator=pseudo_random_generator,
        block_number=block_number
    )
    return iteration.new_state

class InitiatorSetup(NamedTuple):
    current_state: InitiatorPaymentState
    block_number: BlockNumber
    channel: NettingChannelState
    channel_map: Dict[int, NettingChannelState]
    channels: ChannelSet
    available_routes: List[RouteState]
    prng: random.Random
    lock: HashTimeLockState

def setup_initiator_tests(
    amount: TokenAmount = UNIT_TRANSFER_AMOUNT,
    partner_balance: TokenAmount = EMPTY,
    our_address: Address = EMPTY,
    partner_address: Address = EMPTY,
    block_number: BlockNumber = BlockNumber(1),
    allocated_fee: FeeAmount = FeeAmount(0)
) -> InitiatorSetup:
    """Commonly used setup code for initiator manager and channel"""
    prng = random.Random()
    fee_margin = calculate_fee_margin(amount, allocated_fee)
    properties = factories.NettingChannelStateProperties(
        our_state=factories.NettingChannelEndStateProperties(
            balance=amount + allocated_fee + fee_margin,
            address=our_address
        ),
        partner_state=factories.NettingChannelEndStateProperties(
            balance=partner_balance,
            address=partner_address
        )
    )
    channels = factories.make_channel_set([properties])
    transfer_description = create(factories.TransferDescriptionProperties(secret=UNIT_SECRET))
    current_state = make_initiator_manager_state(
        channels=channels,
        transfer_description=transfer_description,
        pseudo_random_generator=prng,
        block_number=block_number,
        estimated_fee=allocated_fee
    )
    initiator_state = get_transfer_at_index(current_state, 0)
    assert initiator_state, 'There should be an initial initiator state'
    lock = channel.get_lock(channels[0].our_state, initiator_state.transfer_description.secrethash)
    assert lock
    available_routes = channels.get_routes(estimated_fee=allocated_fee)
    setup = InitiatorSetup(
        current_state=current_state,
        block_number=block_number,
        channel=channels[0],
        channel_map=channels.channel_map,
        channels=channels,
        available_routes=available_routes,
        prng=prng,
        lock=lock
    )
    return setup

def test_next_route() -> None:
    amount = UNIT_TRANSFER_AMOUNT
    channels = factories.make_channel_set_from_amounts([amount, 0, amount])
    prng = random.Random()
    block_number = BlockNumber(10)
    state = make_initiator_manager_state(channels=channels, pseudo_random_generator=prng, block_number=block_number)
    msg = 'an initialized state must use the first valid route'
    initiator_state = get_transfer_at_index(state, 0)
    assert initiator_state.channel_identifier == channels[0].identifier, msg
    assert not state.cancelled_channels
    initiator_manager.cancel_current_route(payment_state=state, initiator_state=initiator_state)
    assert state.cancelled_channels == [channels[0].identifier]

def test_init_with_usable_routes() -> None:
    transfer_amount = TokenAmount(1000)
    flat_fee = FeeAmount(20)
    expected_balance = calculate_safe_amount_with_fee(transfer_amount, flat_fee)
    properties = factories.NettingChannelStateProperties(
        our_state=factories.NettingChannelEndStateProperties(balance=TokenAmount(expected_balance))
    )
    channels = factories.make_channel_set([properties])
    pseudo_random_generator = random.Random()
    transfer_description = create(TransferDescriptionProperties(secret=UNIT_SECRET, amount=transfer_amount))
    routes = channels.get_routes(estimated_fee=flat_fee)
    init_state_change = ActionInitInitiator(transfer=transfer_description, routes=routes)
    block_number = BlockNumber(1)
    transition = initiator_manager.state_transition(
        payment_state=None,
        state_change=init_state_change,
        channelidentifiers_to_channels=channels.channel_map,
        addresses_to_channel=channels.addresses_to_channel(),
        pseudo_random_generator=pseudo_random_generator,
        block_number=block_number
    )
    assert isinstance(transition.new_state, InitiatorPaymentState)
    assert transition.events, 'we have a valid route, the mediated transfer event must be emitted'
    payment_state = transition.new_state
    assert len(payment_state.routes) == 1
    initiator_state = get_transfer_at_index(payment_state, 0)
    assert initiator_state.transfer_description == transfer_description
    mediated_transfers = [e for e in transition.events if isinstance(e, SendLockedTransfer)]
    assert len(mediated_transfers) == 1, 'mediated_transfer should /not/ split the transfer'
    send_mediated_transfer = mediated_transfers[0]
    transfer = send_mediated_transfer.transfer
    expiration = channel.get_safe_initial_expiration(block_number, channels[0].reveal_timeout)
    assert transfer.balance_proof.token_network_address == channels[0].token_network_address
    assert transfer.balance_proof.locked_amount == expected_balance
    assert transfer.lock.amount == expected_balance
    assert transfer.lock.expiration == expiration
    assert transfer.lock.secrethash == transfer_description.secrethash
    assert send_mediated_transfer.recipient == channels[0].partner_state.address

def test_init_with_fees_more_than_max_limit() -> None:
    transfer_amount = TokenAmount(100)
    flat_fee = FeeAmount(int(transfer_amount + MAX_MEDIATION_FEE_PERC * transfer_amount))
    expected_fee_margin = int(flat_fee * DEFAULT_MEDIATION_FEE_MARGIN)
    properties = factories.NettingChannelStateProperties(
        our_state=factories.NettingChannelEndStateProperties(
            balance=TokenAmount(transfer_amount + flat_fee + expected_fee_margin)
        )
    )
    channels = factories.make_channel_set([properties])
    pseudo_random_generator = random.Random()
    transfer_description = create(TransferDescriptionProperties(secret=UNIT_SECRET, amount=transfer_amount))
    routes = channels.get_routes(estimated_fee=flat_fee)
    init_state_change = ActionInitInitiator(transfer=transfer_description, routes=routes)
    block_number = BlockNumber(1)
    transition = initiator_manager.state_transition(
        payment_state=None,
        state_change=init_state_change,
        channelidentifiers_to_channels=channels.channel_map,
        addresses_to_channel=channels.addresses_to_channel(),
        pseudo_random_generator=pseudo_random_generator,
        block_number=block_number
    )
    assert transition.new_state is None
    assert isinstance(transition.events[0], EventPaymentSentFailed)
    reason_msg = 'None of the available routes could be used and at least one of them exceeded the maximum fee limit (see https://raiden-network.readthedocs.io/en/stable/using-raiden-on-mainnet/overview.html#frequently-asked-questions)'
    assert transition.events[0].reason == reason_msg

def test_init_without_routes() -> None:
    block_number = BlockNumber(1)
    routes: List[RouteState] = []
    pseudo_random_generator = random.Random()
    init_state_change = ActionInitInitiator(factories.UNIT_TRANSFER_DESCRIPTION, routes)
    channel_map: Dict[int, NettingChannelState] = {}
    iteration = initiator_manager.state_transition(
        payment_state=None,
        state_change=init_state_change,
        channelidentifiers_to_channels=channel_map,
        addresses_to_channel={},
        pseudo_random_generator=pseudo_random_generator,
        block_number=block_number
    )
    assert iteration.new_state is None
    assert len(iteration.events) == 1
    assert isinstance(iteration.events[0], EventPaymentSentFailed)
    assert iteration.new_state is None

# ... (rest of the functions with type annotations in similar fashion)

def test_initiator_init() -> None:
    secret = factories.make_secret()
    service = MockRaidenService()
    route_states = [RouteState(route=[factories.make_address(), factories.make_address()], estimated_fee=FeeAmount(123))]
    feedback_token = uuid.uuid4()
    with patch('raiden.routing.get_best_routes', return_value=(None, route_states, feedback_token)) as best_routes:
        assert len(service.route_to_feedback_token) == 0
        _, init_state_change = initiator_init(
            raiden=service,
            transfer_identifier=1,
            transfer_amount=PaymentAmount(100),
            transfer_secret=secret,
            transfer_secrethash=keccak(secret),
            token_network_address=factories.make_token_network_address(),
            target_address=factories.make_address()
        )
        assert best_routes.called
        assert len(service.route_to_feedback_token) == 1
        assert service.route_to_feedback_token[tuple(route_states[0].route)] == feedback_token
        assert init_state_change.routes == route_states

def test_calculate_safe_amount_with_fee() -> None:
    assert calculate_safe_amount_with_fee(1, 0) == 1
    assert calculate_safe_amount_with_fee(1000, 0) == 1000
    assert 1000 <= calculate_safe_amount_with_fee(1000, 1) < 1005
    assert 2000 < calculate_safe_amount_with_fee(1000, 1000) < 2100
