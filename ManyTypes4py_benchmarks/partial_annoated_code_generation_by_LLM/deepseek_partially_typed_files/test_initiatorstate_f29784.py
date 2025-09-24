import random
import uuid
from typing import NamedTuple, Dict, List, Optional, Tuple, Any, cast
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

def get_transfer_at_index(payment_state: InitiatorPaymentState, index: int) -> Optional[InitiatorTransferState]:
    key = list(payment_state.initiator_transfers.keys())[index]
    return payment_state.initiator_transfers[key]

def make_initiator_manager_state(channels: ChannelSet, pseudo_random_generator: random.Random, transfer_description: Optional[factories.TransferDescriptionWithSecretState] = None, block_number: BlockNumber = BlockNumber(1), estimated_fee: FeeAmount = FeeAmount(0)) -> Optional[InitiatorPaymentState]:
    init = ActionInitInitiator(transfer=transfer_description or factories.UNIT_TRANSFER_DESCRIPTION, routes=channels.get_routes(estimated_fee=estimated_fee))
    initial_state = None
    iteration = initiator_manager.state_transition(payment_state=initial_state, state_change=init, channelidentifiers_to_channels=channels.channel_map, addresses_to_channel=channels.addresses_to_channel(), pseudo_random_generator=pseudo_random_generator, block_number=block_number)
    return iteration.new_state

class InitiatorSetup(NamedTuple):
    current_state: State
    block_number: typing.BlockNumber
    channel: NettingChannelState
    channel_map: typing.Dict[typing.ChannelID, NettingChannelState]
    channels: ChannelSet
    available_routes: typing.List[RouteState]
    prng: random.Random
    lock: HashTimeLockState

def setup_initiator_tests(amount: TokenAmount = UNIT_TRANSFER_AMOUNT, partner_balance: TokenAmount = EMPTY, our_address: Address = EMPTY, partner_address: Address = EMPTY, block_number: int = 1, allocated_fee: int = 0) -> InitiatorSetup:
    """Commonly used setup code for initiator manager and channel"""
    prng = random.Random()
    fee_margin = calculate_fee_margin(amount, allocated_fee)
    properties = factories.NettingChannelStateProperties(our_state=factories.NettingChannelEndStateProperties(balance=amount + allocated_fee + fee_margin, address=our_address), partner_state=factories.NettingChannelEndStateProperties(balance=partner_balance, address=partner_address))
    channels = factories.make_channel_set([properties])
    transfer_description = create(TransferDescriptionProperties(secret=UNIT_SECRET))
    current_state = make_initiator_manager_state(channels=channels, transfer_description=transfer_description, pseudo_random_generator=prng, block_number=block_number, estimated_fee=allocated_fee)
    initiator_state = get_transfer_at_index(current_state, 0)
    assert initiator_state, 'There should be an initial initiator state'
    lock = channel.get_lock(channels[0].our_state, initiator_state.transfer_description.secrethash)
    assert lock
    available_routes = channels.get_routes(estimated_fee=allocated_fee)
    setup = InitiatorSetup(current_state=current_state, block_number=block_number, channel=channels[0], channel_map=channels.channel_map, channels=channels, available_routes=available_routes, prng=prng, lock=lock)
    return setup

def test_next_route() -> None:
    amount = UNIT_TRANSFER_AMOUNT
    channels = factories.make_channel_set_from_amounts([amount, 0, amount])
    prng = random.Random()
    block_number = 10
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
    properties = factories.NettingChannelStateProperties(our_state=factories.NettingChannelEndStateProperties(balance=TokenAmount(expected_balance)))
    channels = factories.make_channel_set([properties])
    pseudo_random_generator = random.Random()
    transfer_description = create(TransferDescriptionProperties(secret=UNIT_SECRET, amount=transfer_amount))
    routes = channels.get_routes(estimated_fee=flat_fee)
    init_state_change = ActionInitInitiator(transfer=transfer_description, routes=routes)
    block_number = BlockNumber(1)
    transition = initiator_manager.state_transition(payment_state=None, state_change=init_state_change, channelidentifiers_to_channels=channels.channel_map, addresses_to_channel=channels.addresses_to_channel(), pseudo_random_generator=pseudo_random_generator, block_number=block_number)
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
    properties = factories.NettingChannelStateProperties(our_state=factories.NettingChannelEndStateProperties(balance=TokenAmount(transfer_amount + flat_fee + expected_fee_margin)))
    channels = factories.make_channel_set([properties])
    pseudo_random_generator = random.Random()
    transfer_description = create(TransferDescriptionProperties(secret=UNIT_SECRET, amount=transfer_amount))
    routes = channels.get_routes(estimated_fee=flat_fee)
    init_state_change = ActionInitInitiator(transfer=transfer_description, routes=routes)
    block_number = BlockNumber(1)
    transition = initiator_manager.state_transition(payment_state=None, state_change=init_state_change, channelidentifiers_to_channels=channels.channel_map, addresses_to_channel=channels.addresses_to_channel(), pseudo_random_generator=pseudo_random_generator, block_number=block_number)
    assert transition.new_state is None
    assert isinstance(transition.events[0], EventPaymentSentFailed)
    reason_msg = 'None of the available routes could be used and at least one of them exceeded the maximum fee limit (see https://raiden-network.readthedocs.io/en/stable/using-raiden-on-mainnet/overview.html#frequently-asked-questions)'
    assert transition.events[0].reason == reason_msg

def test_init_without_routes() -> None:
    block_number = BlockNumber(1)
    routes = []
    pseudo_random_generator = random.Random()
    init_state_change = ActionInitInitiator(factories.UNIT_TRANSFER_DESCRIPTION, routes)
    channel_map = {}
    iteration = initiator_manager.state_transition(payment_state=None, state_change=init_state_change, channelidentifiers_to_channels=channel_map, addresses_to_channel={}, pseudo_random_generator=pseudo_random_generator, block_number=block_number)
    assert iteration.new_state is None
    assert len(iteration.events) == 1
    assert isinstance(iteration.events[0], EventPaymentSentFailed)
    assert iteration.new_state is None

def test_state_wait_secretrequest_valid() -> None:
    setup = setup_initiator_tests()
    state_change = ReceiveSecretRequest(payment_identifier=UNIT_TRANSFER_IDENTIFIER, amount=setup.lock.amount, expiration=setup.lock.expiration, secrethash=setup.lock.secrethash, sender=UNIT_TRANSFER_TARGET)
    iteration = initiator_manager.state_transition(payment_state=setup.current_state, state_change=state_change, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    assert len(iteration.events) == 1
    assert isinstance(iteration.events[0], SendSecretReveal)
    initiator_state = get_transfer_at_index(iteration.new_state, 0)
    assert initiator_state.received_secret_request is True
    state_change_2 = ReceiveSecretRequest(payment_identifier=UNIT_TRANSFER_IDENTIFIER, amount=setup.lock.amount, expiration=setup.lock.expiration, secrethash=setup.lock.secrethash, sender=UNIT_TRANSFER_TARGET)
    iteration2 = initiator_manager.state_transition(payment_state=iteration.new_state, state_change=state_change_2, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    assert not iteration2.events

def test_state_wait_secretrequest_invalid_amount() -> None:
    setup = setup_initiator_tests()
    state_change = ReceiveSecretRequest(payment_identifier=UNIT_TRANSFER_IDENTIFIER, amount=setup.lock.amount + 1, expiration=setup.lock.expiration, secrethash=setup.lock.secrethash, sender=UNIT_TRANSFER_TARGET)
    iteration = initiator_manager.state_transition(payment_state=setup.current_state, state_change=state_change, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    msg = 'The payment event now is emitted when the lock expires'
    assert search_for_item(iteration.events, EventPaymentSentFailed, {}) is None, msg
    initiator_state = get_transfer_at_index(iteration.new_state, 0)
    assert initiator_state.received_secret_request is True
    state_change_2 = ReceiveSecretRequest(payment_identifier=UNIT_TRANSFER_IDENTIFIER, amount=setup.lock.amount, expiration=setup.lock.expiration, secrethash=setup.lock.secrethash, sender=UNIT_TRANSFER_TARGET)
    iteration2 = initiator_manager.state_transition(payment_state=iteration.new_state, state_change=state_change_2, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    assert len(iteration2.events) == 0

def test_state_wait_secretrequest_invalid_amount_and_sender() -> None:
    setup = setup_initiator_tests()
    state_change = ReceiveSecretRequest(payment_identifier=UNIT_TRANSFER_IDENTIFIER, amount=setup.lock.amount + 1, expiration=setup.lock.expiration, secrethash=setup.lock.secrethash, sender=UNIT_TRANSFER_INITIATOR)
    iteration = initiator_manager.state_transition(payment_state=setup.current_state, state_change=state_change, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    assert len(iteration.events) == 0
    initiator_state = get_transfer_at_index(iteration.new_state, 0)
    assert initiator_state.received_secret_request is False
    state_change_2 = ReceiveSecretRequest(payment_identifier=UNIT_TRANSFER_IDENTIFIER, amount=setup.lock.amount, expiration=setup.lock.expiration, secrethash=setup.lock.secrethash, sender=UNIT_TRANSFER_TARGET)
    iteration2 = initiator_manager.state_transition(payment_state=iteration.new_state, state_change=state_change_2, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    initiator_state = get_transfer_at_index(iteration2.new_state, 0)
    assert initiator_state.received_secret_request is True
    assert isinstance(iteration2.events[0], SendSecretReveal)

def test_state_wait_unlock_valid() -> None:
    setup = setup_initiator_tests()
    initiator_state = get_transfer_at_index(setup.current_state, 0)
    initiator_state.transfer_state = 'transfer_secret_revealed'
    state_change = ReceiveSecretReveal(secret=UNIT_SECRET, sender=setup.channel.partner_state.address)
    iteration = initiator_manager.state_transition(payment_state=setup.current_state, state_change=state_change, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    assert len(iteration.events) == 3
    balance_proof = search_for_item(iteration.events, SendUnlock, {})
    complete = search_for_item(iteration.events, EventPaymentSentSuccess, {})
    assert search_for_item(iteration.events, EventUnlockSuccess, {})
    assert balance_proof
    assert complete
    assert len(complete.route) == 2
    assert complete.route[1] == balance_proof.recipient
    assert balance_proof.recipient == setup.channel.partner_state.address
    assert complete.identifier == UNIT_TRANSFER_IDENTIFIER
    assert iteration.new_state is None, 'state must be cleaned'

def test_state_wait_unlock_invalid() -> None:
    setup = setup_initiator_tests()
    initiator_state = get_transfer_at_index(setup.current_state, 0)
    initiator_state.transfer_state = 'transfer_secret_revealed'
    before_state = deepcopy(setup.current_state)
    state_change = ReceiveSecretReveal(secret=UNIT_SECRET, sender=factories.ADDR)
    iteration = initiator_manager.state_transition(payment_state=setup.current_state, state_change=state_change, channelidentifiers_to_channels=setup.channel_map, addresses_to_channel=setup.channels.addresses_to_channel(), pseudo_random_generator=setup.prng, block_number=setup.block_number)
    assert not iteration.events
    assert iteration.new_state == before_state

def channels_setup(amount: TokenAmount, our_address: Address, refund_address: Address) -> ChannelSet:
    funded = factories.NettingChannelEndStateProperties(balance=amount, address=our_address)
    broke = factories.replace(funded, balance=0)
    funded_partner = factories.replace(funded, address=refund_address)
    properties = [factories.NettingChannelStateProperties(our_state=funded, partner_state=funded_partner), factories.NettingChannelStateProperties(our_state=broke), factories.NettingChannelStateProperties(our_state=funded)]
    return factories.make_channel_set(properties)

def test_ref