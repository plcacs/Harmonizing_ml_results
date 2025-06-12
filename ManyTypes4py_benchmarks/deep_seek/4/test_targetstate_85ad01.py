import random
from typing import NamedTuple, Optional, List, Dict, Any, Tuple, cast
import pytest
from raiden.constants import EMPTY_SECRET, EMPTY_SECRET_SHA256, LOCKSROOT_OF_NO_LOCKS, UINT64_MAX
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.tests.utils import factories
from raiden.tests.utils.events import search_for_item
from raiden.tests.utils.factories import UNIT_SECRET, UNIT_SECRETHASH, UNIT_TRANSFER_INITIATOR, UNIT_TRANSFER_SENDER, UNIT_TRANSFER_TARGET, BalanceProofSignedStateProperties, LockedTransferSignedStateProperties, NettingChannelEndStateProperties, NettingChannelStateProperties, create, make_channel_set
from raiden.transfer import channel
from raiden.transfer.events import ContractSendSecretReveal, SendProcessed
from raiden.transfer.mediated_transfer import target
from raiden.transfer.mediated_transfer.events import EventUnlockClaimFailed, SendSecretRequest, SendSecretReveal
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitTarget, ReceiveLockExpired, ReceiveSecretReveal
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, ReceiveUnlock
from raiden.utils import typing
from raiden.utils.typing import TokenAmount, Address, BlockNumber, BlockHash, Secret, SecretHash, PaymentID, Nonce, Locksroot, MessageID

def make_target_transfer(
    channel: NettingChannelStateProperties,
    amount: Optional[TokenAmount] = None,
    expiration: Optional[BlockNumber] = None,
    initiator: Optional[Address] = None,
    block_number: BlockNumber = 1
) -> LockedTransferSignedStateProperties:
    default_expiration = block_number + channel.settle_timeout - channel.reveal_timeout
    return factories.make_signed_transfer_for(
        channel,
        factories.LockedTransferSignedStateProperties(
            amount=amount or channel.partner_state.contract_balance,
            expiration=expiration or default_expiration,
            initiator=initiator or UNIT_TRANSFER_INITIATOR,
            target=channel.our_state.address
        )
    )

class TargetStateSetup(NamedTuple):
    channel: NettingChannelStateProperties
    new_state: TargetTransferState
    our_address: Address
    initiator: Address
    expiration: BlockNumber
    amount: TokenAmount
    block_number: BlockNumber
    pseudo_random_generator: random.Random

def make_target_state(
    our_address: Address = factories.ADDR,
    amount: TokenAmount = 3,
    block_number: BlockNumber = 1,
    initiator: Address = UNIT_TRANSFER_INITIATOR,
    expiration: Optional[BlockNumber] = None,
    pseudo_random_generator: Optional[random.Random] = None
) -> TargetStateSetup:
    pseudo_random_generator = pseudo_random_generator or random.Random()
    channels = make_channel_set([
        NettingChannelStateProperties(
            our_state=NettingChannelEndStateProperties(address=our_address),
            partner_state=NettingChannelEndStateProperties(address=UNIT_TRANSFER_SENDER, balance=amount)
        )
    ])
    expiration = expiration or channels[0].reveal_timeout + block_number + 1
    from_transfer = make_target_transfer(channels[0], amount, expiration, initiator)
    state_change = ActionInitTarget(
        from_hop=channels.get_hop(0),
        transfer=from_transfer,
        balance_proof=from_transfer.balance_proof,
        sender=from_transfer.balance_proof.sender
    )
    iteration = target.handle_inittarget(
        state_change=state_change,
        channel_state=channels[0],
        pseudo_random_generator=pseudo_random_generator,
        block_number=block_number
    )
    return TargetStateSetup(
        channel=channels[0],
        new_state=iteration.new_state,
        our_address=our_address,
        initiator=initiator,
        expiration=expiration,
        amount=amount,
        block_number=block_number,
        pseudo_random_generator=pseudo_random_generator
    )

channel_properties = NettingChannelStateProperties(
    our_state=NettingChannelEndStateProperties(address=UNIT_TRANSFER_TARGET),
    partner_state=NettingChannelEndStateProperties(address=UNIT_TRANSFER_SENDER, balance=TokenAmount(3))
)

channel_properties2 = NettingChannelStateProperties(
    our_state=NettingChannelEndStateProperties(address=factories.make_address(), balance=TokenAmount(100)),
    partner_state=NettingChannelEndStateProperties(address=UNIT_TRANSFER_SENDER, balance=TokenAmount(130))
)

def test_events_for_onchain_secretreveal() -> None:
    block_number = 10
    expiration = block_number + 30
    channels = make_channel_set([channel_properties])
    from_transfer = make_target_transfer(channels[0], expiration=expiration)
    channel.handle_receive_lockedtransfer(channels[0], from_transfer)
    channel.register_offchain_secret(channels[0], UNIT_SECRET, UNIT_SECRETHASH)
    safe_to_wait = expiration - channels[0].reveal_timeout - 1
    unsafe_to_wait = expiration - channels[0].reveal_timeout
    state = TargetTransferState(channels.get_hop(0), from_transfer)
    events = target.events_for_onchain_secretreveal(
        target_state=state,
        channel_state=channels[0],
        block_number=safe_to_wait,
        block_hash=factories.make_block_hash()
    )
    assert not events
    events = target.events_for_onchain_secretreveal(
        target_state=state,
        channel_state=channels[0],
        block_number=unsafe_to_wait,
        block_hash=factories.make_block_hash()
    )
    msg = 'when its not safe to wait, the contract send must be emitted'
    assert search_for_item(events, ContractSendSecretReveal, {'secret': UNIT_SECRET}), msg
    msg = 'second call must not emit ContractSendSecretReveal again'
    assert not target.events_for_onchain_secretreveal(
        target_state=state,
        channel_state=channels[0],
        block_number=unsafe_to_wait,
        block_hash=factories.make_block_hash()
    ), msg

def test_handle_inittarget() -> None:
    block_number = 1
    pseudo_random_generator = random.Random()
    channels = make_channel_set([channel_properties])
    transfer_properties = LockedTransferSignedStateProperties(
        amount=channels[0].partner_state.contract_balance,
        expiration=channels[0].reveal_timeout + block_number + 1,
        canonical_identifier=channels[0].canonical_identifier,
        transferred_amount=0,
        locked_amount=channels[0].partner_state.contract_balance
    )
    from_transfer = create(transfer_properties)
    state_change = ActionInitTarget(
        from_hop=channels.get_hop(0),
        transfer=from_transfer,
        balance_proof=from_transfer.balance_proof,
        sender=from_transfer.balance_proof.sender
    )
    iteration = target.handle_inittarget(state_change, channels[0], pseudo_random_generator, block_number)
    assert search_for_item(
        iteration.events,
        SendSecretRequest,
        {
            'payment_identifier': from_transfer.payment_identifier,
            'amount': from_transfer.lock.amount,
            'secrethash': from_transfer.lock.secrethash,
            'recipient': UNIT_TRANSFER_INITIATOR
        }
    )
    assert search_for_item(iteration.events, SendProcessed, {})

def test_handle_inittarget_bad_expiration() -> None:
    block_number = 1
    pseudo_random_generator = random.Random()
    channels = make_channel_set([channel_properties])
    expiration = channels[0].reveal_timeout + block_number + 1
    from_transfer = make_target_transfer(channels[0], expiration=expiration)
    channel.handle_receive_lockedtransfer(channels[0], from_transfer)
    state_change = ActionInitTarget(
        from_hop=channels.get_hop(0),
        transfer=from_transfer,
        balance_proof=from_transfer.balance_proof,
        sender=from_transfer.balance_proof.sender
    )
    iteration = target.handle_inittarget(state_change, channels[0], pseudo_random_generator, block_number)
    assert search_for_item(iteration.events, EventUnlockClaimFailed, {})

def test_handle_offchain_secretreveal() -> None:
    setup = make_target_state()
    state_change = ReceiveSecretReveal(secret=UNIT_SECRET, sender=setup.initiator)
    iteration = target.handle_offchain_secretreveal(
        target_state=setup.new_state,
        state_change=state_change,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=setup.block_number
    )
    assert len(iteration.events) == 1
    reveal = iteration.events[0]
    assert isinstance(reveal, SendSecretReveal)
    assert iteration.new_state.state == 'reveal_secret'
    assert reveal.secret == UNIT_SECRET
    assert reveal.recipient == setup.new_state.from_hop.node_address
    secret = EMPTY_SECRET
    state_change = ReceiveSecretReveal(secret, setup.initiator)
    iteration = target.handle_offchain_secretreveal(
        target_state=setup.new_state,
        state_change=state_change,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=setup.block_number
    )
    assert len(iteration.events) == 0

def test_handle_offchain_secretreveal_after_lock_expired() -> None:
    setup = make_target_state()
    lock_expiration_block_number = channel.get_sender_expiration_threshold(setup.new_state.transfer.lock.expiration)
    lock_expiration_block = Block(
        block_number=lock_expiration_block_number,
        gas_limit=1,
        block_hash=factories.make_transaction_hash()
    )
    iteration = target.state_transition(
        target_state=setup.new_state,
        state_change=lock_expiration_block,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=lock_expiration_block_number
    )
    state = iteration.new_state
    msg = 'At the expiration block we should get an EventUnlockClaimFailed'
    assert search_for_item(iteration.events, EventUnlockClaimFailed, {}), msg
    iteration = target.state_transition(
        target_state=state,
        state_change=ReceiveSecretReveal(UNIT_SECRET, setup.initiator),
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=lock_expiration_block_number + 1
    )
    state = iteration.new_state
    next_block = Block(
        block_number=lock_expiration_block_number + 1,
        gas_limit=1,
        block_hash=factories.make_transaction_hash()
    )
    iteration = target.state_transition(
        target_state=state,
        state_change=next_block,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=lock_expiration_block_number + 1
    )
    msg = 'At the next block we should not get the same event'
    assert not search_for_item(iteration.events, EventUnlockClaimFailed, {}), msg

def test_handle_onchain_secretreveal() -> None:
    setup = make_target_state(block_number=1, expiration=1 + factories.UNIT_REVEAL_TIMEOUT)
    assert factories.UNIT_SECRETHASH in setup.channel.partner_state.secrethashes_to_lockedlocks
    offchain_secret_reveal_iteration = target.state_transition(
        target_state=setup.new_state,
        state_change=ReceiveSecretReveal(secret=UNIT_SECRET, sender=setup.initiator),
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=setup.block_number
    )
    assert UNIT_SECRETHASH in setup.channel.partner_state.secrethashes_to_unlockedlocks
    assert UNIT_SECRETHASH not in setup.channel.partner_state.secrethashes_to_lockedlocks
    block_number_prior_the_expiration = setup.expiration - 2
    onchain_reveal = ContractReceiveSecretReveal(
        transaction_hash=factories.make_address(),
        secret_registry_address=factories.make_address(),
        secrethash=EMPTY_SECRET_SHA256,
        secret=EMPTY_SECRET,
        block_number=block_number_prior_the_expiration,
        block_hash=factories.make_block_hash()
    )
    onchain_secret_reveal_iteration = target.state_transition(
        target_state=offchain_secret_reveal_iteration.new_state,
        state_change=onchain_reveal,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=block_number_prior_the_expiration
    )
    unlocked_onchain = setup.channel.partner_state.secrethashes_to_onchain_unlockedlocks
    assert EMPTY_SECRET_SHA256 not in unlocked_onchain
    onchain_reveal = ContractReceiveSecretReveal(
        transaction_hash=factories.make_address(),
        secret_registry_address=factories.make_address(),
        secrethash=UNIT_SECRETHASH,
        secret=UNIT_SECRET,
        block_number=block_number_prior_the_expiration,
        block_hash=factories.make_block_hash()
    )
    onchain_secret_reveal_iteration = target.state_transition(
        target_state=offchain_secret_reveal_iteration.new_state,
        state_change=onchain_reveal,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=block_number_prior_the_expiration
    )
    unlocked_onchain = setup.channel.partner_state.secrethashes_to_onchain_unlockedlocks
    assert UNIT_SECRETHASH in unlocked_onchain
    extra_block_handle_transition = target.handle_block(
        target_state=onchain_secret_reveal_iteration.new_state,
        channel_state=setup.channel,
        block_number=block_number_prior_the_expiration + 1,
        block_hash=factories.make_block_hash()
    )
    assert len(extra_block_handle_transition.events) == 0

def test_handle_block() -> None:
    setup = make_target_state()
    new_block = Block(
        block_number=setup.block_number + 1,
        gas_limit=1,
        block_hash=factories.make_transaction_hash()
    )
    iteration = target.state_transition(
        target_state=setup.new_state,
        state_change=new_block,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=new_block.block_number
    )
    assert iteration.new_state
    assert not iteration.events

def test_handle_block_equal_block_number() -> None:
    setup = make_target_state()
    new_block = Block(
        block_number=1,
        gas_limit=1,
        block_hash=factories.make_transaction_hash()
    )
    iteration = target.state_transition(
        target_state=setup.new_state,
        state_change=new_block,
        channel_state=setup.channel,
        pseudo_random_generator=random.Random(),
        block_number=new_block.block_number
    )
    assert iteration.new_state
    assert not iteration.events

def test_handle_block_lower_block_number() -> None:
    setup = make_target_state(block_number=10)
    new_block = Block(
        block_number=setup.block_number - 1,
        gas_limit=1,
        block_hash=factories.make_transaction_hash()
    )
    iteration = target.state_transition(
        target_state=setup.new_state,
        state_change=new_block,
        channel_state=setup.channel,
        pseudo_random_generator=setup.pseudo_random_generator,
        block_number=new_block.block_number
    )
    assert iteration.new_state
    assert not iteration.events

def test_state_transition() -> None:
    lock_amount = 7
    block_number = 1
    initiator = factories.make_address()
    pseudo_random_generator = random.Random()
    channels = make_channel_set([channel_properties2])
    from_transfer = make_target_transfer(channels[0], amount=lock_amount, initiator=initiator)
    init = ActionInitTarget(
        from_hop=channels.get_hop(0),
        transfer=from_transfer,
        balance_proof=from_transfer.balance_proof,
        sender=from_transfer.balance_proof.sender
    )
    init_transition = target.state_transition(
        target_state=None,
        state_change=init,
        channel_state=channels[0],
        pseudo_random_generator=pseudo_random_generator,
        block_number=block_number
    )
    assert init_transition.new_state is not None
    assert init_transition.new_state.from_hop == channels.get_hop(0)
    assert init_transition.new_state.transfer == from_transfer
    first_new_block = Block(
        block_number=block_number + 1,
        gas_limit=1,
        block_hash=factories.make_transaction_hash()
    )
    first_block_iteration = target.state_transition(
        target_state=init_transition.new_state,
        state_change=first_new_block,
        channel_state=channels[0],
        pseudo_random_generator=pseudo_random_generator,
        block_number=first_new_block.block_number
    )
    secret_reveal = ReceiveSecretReveal(secret=UNIT_SECRET, sender=initiator)
    reveal_iteration = target.state_transition(
        target_state=first_block_iteration.new_state,
        state_change=secret_reveal,
        channel_state=channels[0],
        pseudo_random_generator=pseudo_random_generator,
        block_number=first_new_block.block_number
    )
    assert reveal_iteration.events
    second_new_block = Block(
        block_number=block_number + 2,
        gas_limit=1,
        block