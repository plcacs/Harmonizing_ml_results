import random
from math import ceil
from typing import Tuple
from raiden.constants import ABSENT_SECRET
from raiden.settings import DEFAULT_MEDIATION_FEE_MARGIN, DEFAULT_WAIT_BEFORE_LOCK_REMOVAL, MAX_MEDIATION_FEE_PERC, PAYMENT_AMOUNT_BASED_FEE_MARGIN
from raiden.transfer import channel
from raiden.transfer.architecture import Event, TransitionResult
from raiden.transfer.events import EventInvalidSecretRequest, EventPaymentSentFailed, EventPaymentSentSuccess
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE
from raiden.transfer.mediated_transfer.events import EventRouteFailed, EventUnlockFailed, EventUnlockSuccess, SendLockedTransfer, SendSecretReveal
from raiden.transfer.mediated_transfer.state import InitiatorTransferState, TransferDescriptionWithSecretState
from raiden.transfer.mediated_transfer.state_change import ReceiveSecretRequest, ReceiveSecretReveal
from raiden.transfer.state import ChannelState, NettingChannelState, RouteState, get_address_metadata, message_identifier_from_prng
from raiden.transfer.state_change import Block, ContractReceiveSecretReveal, StateChange
from raiden.transfer.utils import is_valid_secret_reveal
from raiden.utils.typing import MYPY_ANNOTATION, Address, BlockExpiration, BlockNumber, Dict, FeeAmount, List, MessageID, Optional, PaymentAmount, PaymentWithFeeAmount, Secret, SecretHash, TokenNetworkAddress

def calculate_fee_margin(payment_amount: Union[raiden.utils.PaymentAmount, raiden.utils.FeeAmount, int], estimated_fee: Union[raiden.utils.PaymentAmount, raiden.utils.FeeAmount, int]) -> FeeAmount:
    if estimated_fee == 0:
        return FeeAmount(0)
    return FeeAmount(int(ceil(abs(estimated_fee) * DEFAULT_MEDIATION_FEE_MARGIN + payment_amount * PAYMENT_AMOUNT_BASED_FEE_MARGIN)))

def calculate_safe_amount_with_fee(payment_amount: Union[raiden.utils.PaymentAmount, raiden.utils.FeeAmount, raiden.utils.PaymentWithFeeAmount], estimated_fee: Union[raiden.utils.PaymentAmount, raiden.utils.FeeAmount, raiden.utils.PaymentWithFeeAmount]) -> PaymentWithFeeAmount:
    """Calculates the total payment amount

    This total amount consists of the payment amount, the estimated fees as well as a
    small margin that is added to increase the likelihood of payments succeeding in
    conditions where channels are used for multiple payments.

    We could get much better margins by considering that we only need margins
    for imbalance fees. See
    https://github.com/raiden-network/raiden-services/issues/569.
    """
    return PaymentWithFeeAmount(payment_amount + estimated_fee + calculate_fee_margin(payment_amount, estimated_fee))

def events_for_unlock_lock(initiator_state: Union[raiden.transfer.state.NettingChannelState, random.Random, raiden.utils.BlockNumber], channel_state: Union[raiden.transfer.state.NettingChannelState, raiden.transfer.mediated_transfer.state.InitiatorTransferState, raiden.utils.BlockNumber], secret: Union[raiden.utils.BlockNumber, raiden.utils.Secret, raiden.utils.SecretHash], secrethash: Union[raiden.utils.BlockNumber, raiden.utils.Secret, raiden.utils.SecretHash], pseudo_random_generator: Union[random.Random, int, raiden.utils.Address], block_number: Union[raiden.utils.BlockNumber, raiden.utils.Secret, raiden.utils.SecretHash]) -> list[typing.Union[EventPaymentSentSuccess,EventUnlockSuccess]]:
    """Unlocks the lock offchain, and emits the events for the successful payment."""
    transfer_description = initiator_state.transfer_description
    message_identifier = message_identifier_from_prng(pseudo_random_generator)
    recipient_address = channel_state.partner_state.address
    recipient_metadata = get_address_metadata(recipient_address, [initiator_state.route])
    unlock_lock = channel.send_unlock(channel_state=channel_state, message_identifier=message_identifier, payment_identifier=transfer_description.payment_identifier, secret=secret, secrethash=secrethash, block_number=block_number, recipient_metadata=recipient_metadata)
    payment_sent_success = EventPaymentSentSuccess(token_network_registry_address=channel_state.token_network_registry_address, token_network_address=channel_state.token_network_address, identifier=transfer_description.payment_identifier, amount=transfer_description.amount, target=transfer_description.target, secret=secret, route=initiator_state.route.route)
    unlock_success = EventUnlockSuccess(transfer_description.payment_identifier, transfer_description.secrethash)
    return [unlock_lock, payment_sent_success, unlock_success]

def handle_block(initiator_state: Union[raiden.transfer.mediated_transfer.state.InitiatorTransferState, random.Random, raiden.transfer.state.NettingChannelState], state_change: Union[raiden.transfer.state_change.Block, random.Random, raiden.utils.BlockHash], channel_state: Union[raiden.transfer.state.NettingChannelState, random.Random, raiden.utils.BlockNumber], pseudo_random_generator: Union[random.Random, raiden.utils.ChannelMap, raiden.utils.BlockNumber]) -> TransitionResult:
    """Checks if the lock has expired, and if it has sends a remove expired
    lock and emits the failing events.
    """
    secrethash = initiator_state.transfer.lock.secrethash
    locked_lock = channel_state.our_state.secrethashes_to_lockedlocks.get(secrethash)
    if not locked_lock:
        if channel_state.partner_state.secrethashes_to_lockedlocks.get(secrethash):
            return TransitionResult(initiator_state, [])
        else:
            return TransitionResult(None, [])
    lock_expiration_threshold = BlockExpiration(locked_lock.expiration + DEFAULT_WAIT_BEFORE_LOCK_REMOVAL)
    lock_has_expired = channel.is_lock_expired(end_state=channel_state.our_state, lock=locked_lock, block_number=state_change.block_number, lock_expiration_threshold=lock_expiration_threshold)
    events = []
    if lock_has_expired and initiator_state.transfer_state != 'transfer_expired':
        is_channel_open = channel.get_status(channel_state) == ChannelState.STATE_OPENED
        if is_channel_open:
            recipient_address = channel_state.partner_state.address
            recipient_metadata = get_address_metadata(recipient_address, [initiator_state.route])
            expired_lock_events = channel.send_lock_expired(channel_state=channel_state, locked_lock=locked_lock, pseudo_random_generator=pseudo_random_generator, recipient_metadata=recipient_metadata)
            events.extend(expired_lock_events)
        if initiator_state.received_secret_request:
            reason = 'lock expired, despite receiving secret request'
        else:
            reason = 'lock expired'
        transfer_description = initiator_state.transfer_description
        payment_identifier = transfer_description.payment_identifier
        payment_failed = EventPaymentSentFailed(token_network_registry_address=transfer_description.token_network_registry_address, token_network_address=transfer_description.token_network_address, identifier=payment_identifier, target=transfer_description.target, reason=reason)
        route_failed = EventRouteFailed(secrethash=secrethash, route=initiator_state.route.route, token_network_address=transfer_description.token_network_address)
        unlock_failed = EventUnlockFailed(identifier=payment_identifier, secrethash=initiator_state.transfer_description.secrethash, reason=reason)
        lock_exists = channel.lock_exists_in_either_channel_side(channel_state=channel_state, secrethash=secrethash)
        initiator_state.transfer_state = 'transfer_expired'
        return TransitionResult(initiator_state if lock_exists else None, events + [payment_failed, route_failed, unlock_failed])
    else:
        return TransitionResult(initiator_state, events)

def try_new_route(addresses_to_channel: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState], candidate_route_states: Union[raiden.utils.BlockNumber, raiden.utils.Address, raiden.transfer.mediated_transfer.state.LockedTransferSignedState], transfer_description: Union[raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, random.Random], pseudo_random_generator: Union[random.Random, int, raiden.utils.BlockNumber], block_number: Union[raiden.utils.BlockNumber, raiden.utils.Address, raiden.transfer.mediated_transfer.state.LockedTransferSignedState]) -> TransitionResult:
    initiator_state = None
    events = []
    route_fee_exceeds_max = False
    channel_state = None
    chosen_route_state = None
    try:
        any_channel = next(iter(addresses_to_channel.values()))
        our_address = any_channel.our_state.address
    except StopIteration:
        pass
    else:
        for route_state in candidate_route_states:
            next_hop = route_state.hop_after(our_address)
            if not next_hop:
                continue
            candidate_channel_state = addresses_to_channel.get((transfer_description.token_network_address, next_hop))
            if candidate_channel_state is None:
                continue
            amount_with_fee = calculate_safe_amount_with_fee(payment_amount=transfer_description.amount, estimated_fee=route_state.estimated_fee)
            max_amount_limit = transfer_description.amount + int(transfer_description.amount * MAX_MEDIATION_FEE_PERC)
            if amount_with_fee > max_amount_limit:
                route_fee_exceeds_max = True
                continue
            channel_usability_state = channel.is_channel_usable_for_new_transfer(channel_state=candidate_channel_state, transfer_amount=amount_with_fee, lock_timeout=transfer_description.lock_timeout)
            if channel_usability_state is channel.ChannelUsability.USABLE:
                channel_state = candidate_channel_state
                chosen_route_state = route_state
                break
    if chosen_route_state is None:
        reason = 'None of the available routes could be used'
        if route_fee_exceeds_max:
            reason += ' and at least one of them exceeded the maximum fee limit (see https://raiden-network.readthedocs.io/en/stable/using-raiden-on-mainnet/overview.html#frequently-asked-questions)'
        transfer_failed = EventPaymentSentFailed(token_network_registry_address=transfer_description.token_network_registry_address, token_network_address=transfer_description.token_network_address, identifier=transfer_description.payment_identifier, target=transfer_description.target, reason=reason)
        events.append(transfer_failed)
        initiator_state = None
    else:
        assert channel_state is not None, 'We must have a channel_state if we have a route_state'
        message_identifier = message_identifier_from_prng(pseudo_random_generator)
        lockedtransfer_event = send_lockedtransfer(transfer_description=transfer_description, channel_state=channel_state, message_identifier=message_identifier, block_number=block_number, route_state=chosen_route_state, route_states=candidate_route_states)
        initiator_state = InitiatorTransferState(route=chosen_route_state, transfer_description=transfer_description, channel_identifier=channel_state.identifier, transfer=lockedtransfer_event.transfer)
        events.append(lockedtransfer_event)
    return TransitionResult(initiator_state, events)

def send_lockedtransfer(transfer_description: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.transfer.state.NettingChannelState, raiden.utils.MessageID], channel_state: Union[raiden.transfer.mediated_transfer.state.TransferDescriptionWithSecretState, raiden.transfer.state.NettingChannelState, raiden.utils.MessageID], message_identifier: Union[raiden.utils.MessageID, raiden.utils.InitiatorAddress, raiden.utils.PaymentAmount], block_number: Union[raiden.utils.BlockNumber, float], route_state: raiden.utils.TokenAmount, route_states: Union[raiden.utils.TokenAddress, raiden.utils.PaymentID]):
    """Create a mediated transfer using channel."""
    assert channel_state.token_network_address == transfer_description.token_network_address, 'token_network_address mismatch'
    lock_expiration = channel.get_safe_initial_expiration(block_number, channel_state.reveal_timeout, transfer_description.lock_timeout)
    total_amount = calculate_safe_amount_with_fee(payment_amount=transfer_description.amount, estimated_fee=route_state.estimated_fee)
    recipient_address = channel_state.partner_state.address
    recipient_metadata = get_address_metadata(recipient_address, route_states)
    lockedtransfer_event = channel.send_lockedtransfer(channel_state=channel_state, initiator=transfer_description.initiator, target=transfer_description.target, amount=total_amount, message_identifier=message_identifier, payment_identifier=transfer_description.payment_identifier, expiration=lock_expiration, secret=transfer_description.secret, secrethash=transfer_description.secrethash, route_states=route_states, recipient_metadata=recipient_metadata, previous_metadata=None)
    return lockedtransfer_event

def handle_secretrequest(initiator_state: Union[raiden.transfer.mediated_transfer.state.InitiatorTransferState, raiden.transfer.mediated_transfer.state_change.ReceiveSecretRequest, random.Random], state_change: Union[raiden.transfer.mediated_transfer.state.InitiatorTransferState, raiden.transfer.mediated_transfer.state_change.ReceiveSecretRequest, random.Random], channel_state: Union[raiden.transfer.state.NettingChannelState, random.Random, raiden.transfer.mediated_transfer.state.InitiatorTransferState], pseudo_random_generator: Union[random.Random, bytes, raiden.transfer.state_change.Block]) -> TransitionResult:
    is_message_from_target = state_change.sender == Address(initiator_state.transfer_description.target) and state_change.secrethash == initiator_state.transfer_description.secrethash and (state_change.payment_identifier == initiator_state.transfer_description.payment_identifier)
    if not is_message_from_target:
        return TransitionResult(initiator_state, [])
    lock = channel.get_lock(channel_state.our_state, initiator_state.transfer_description.secrethash)
    assert lock is not None, "channel is does not have the transfer's lock"
    if initiator_state.received_secret_request:
        return TransitionResult(initiator_state, [])
    is_valid_secretrequest = state_change.amount >= initiator_state.transfer_description.amount and state_change.expiration == lock.expiration and (initiator_state.transfer_description.secret != ABSENT_SECRET)
    if is_valid_secretrequest:
        message_identifier = message_identifier_from_prng(pseudo_random_generator)
        transfer_description = initiator_state.transfer_description
        recipient = Address(transfer_description.target)
        recipient_metadata = get_address_metadata(recipient, [initiator_state.route])
        revealsecret = SendSecretReveal(recipient=recipient, recipient_metadata=recipient_metadata, message_identifier=message_identifier, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE, secret=transfer_description.secret)
        initiator_state.transfer_state = 'transfer_secret_revealed'
        initiator_state.received_secret_request = True
        return TransitionResult(initiator_state, [revealsecret])
    else:
        initiator_state.received_secret_request = True
        invalid_request = EventInvalidSecretRequest(payment_identifier=state_change.payment_identifier, intended_amount=initiator_state.transfer_description.amount, actual_amount=state_change.amount)
        return TransitionResult(initiator_state, [invalid_request])

def handle_offchain_secretreveal(initiator_state: Union[raiden.transfer.mediated_transfer.state.InitiatorTransferState, raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal, raiden.utils.BlockNumber], state_change: Union[raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal, raiden.transfer.mediated_transfer.state.InitiatorTransferState, random.Random], channel_state: Union[raiden.transfer.state.NettingChannelState, random.Random, raiden.utils.BlockNumber], pseudo_random_generator: Union[random.Random, raiden.transfer.state.NettingChannelState, raiden.utils.BlockNumber], block_number: Union[raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state.InitiatorTransferState, raiden.transfer.state.NettingChannelState]) -> TransitionResult:
    """Once the next hop proves it knows the secret, the initiator can unlock
    the mediated transfer.

    This will validate the secret, and if valid a new balance proof is sent to
    the next hop with the current lock removed from the pending locks and the
    transferred amount updated.
    """
    valid_reveal = is_valid_secret_reveal(state_change=state_change, transfer_secrethash=initiator_state.transfer_description.secrethash)
    sent_by_partner = state_change.sender == channel_state.partner_state.address
    is_channel_open = channel.get_status(channel_state) == ChannelState.STATE_OPENED
    lock = initiator_state.transfer.lock
    expired = channel.is_lock_expired(end_state=channel_state.our_state, lock=lock, block_number=block_number, lock_expiration_threshold=lock.expiration)
    if valid_reveal and is_channel_open and sent_by_partner and (not expired):
        events = events_for_unlock_lock(initiator_state=initiator_state, channel_state=channel_state, secret=state_change.secret, secrethash=state_change.secrethash, pseudo_random_generator=pseudo_random_generator, block_number=block_number)
        iteration = TransitionResult(None, events)
    else:
        events = []
        iteration = TransitionResult(initiator_state, events)
    return iteration

def handle_onchain_secretreveal(initiator_state: Union[raiden.transfer.mediated_transfer.state.InitiatorTransferState, raiden.utils.BlockNumber, random.Random], state_change: Union[raiden.transfer.state_change.ContractReceiveSecretReveal, raiden.transfer.mediated_transfer.state.InitiatorTransferState, raiden.utils.BlockNumber], channel_state: Union[raiden.transfer.state.NettingChannelState, random.Random, raiden.utils.BlockNumber], pseudo_random_generator: Union[random.Random, raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state_change.ReceiveSecretReveal], block_number: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState]) -> TransitionResult:
    """When a secret is revealed on-chain all nodes learn the secret.

    This check the on-chain secret corresponds to the one used by the
    initiator, and if valid a new balance proof is sent to the next hop with
    the current lock removed from the pending locks and the transferred amount
    updated.
    """
    secret = state_change.secret
    secrethash = initiator_state.transfer_description.secrethash
    is_valid_secret = is_valid_secret_reveal(state_change=state_change, transfer_secrethash=secrethash)
    is_channel_open = channel.get_status(channel_state) == ChannelState.STATE_OPENED
    is_lock_expired = state_change.block_number > initiator_state.transfer.lock.expiration
    is_lock_unlocked = is_valid_secret and (not is_lock_expired)
    if is_lock_unlocked:
        channel.register_onchain_secret(channel_state=channel_state, secret=secret, secrethash=secrethash, secret_reveal_block_number=state_change.block_number)
    lock = initiator_state.transfer.lock
    expired = channel.is_lock_expired(end_state=channel_state.our_state, lock=lock, block_number=block_number, lock_expiration_threshold=lock.expiration)
    if is_lock_unlocked and is_channel_open and (not expired):
        events = events_for_unlock_lock(initiator_state, channel_state, state_change.secret, state_change.secrethash, pseudo_random_generator, block_number)
        iteration = TransitionResult(None, events)
    else:
        events = []
        iteration = TransitionResult(initiator_state, events)
    return iteration

def state_transition(initiator_state: Union[random.Random, raiden.transfer.state.NettingChannelState, raiden.transfer.mediated_transfer.state.InitiatorTransferState], state_change: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.mediated_transfer.state.MediatorTransferState], channel_state: Union[random.Random, raiden.transfer.state.NettingChannelState, raiden.transfer.mediated_transfer.state.InitiatorTransferState], pseudo_random_generator: Union[random.Random, raiden.transfer.state.NettingChannelState, raiden.transfer.mediated_transfer.state.InitiatorTransferState], block_number: Union[raiden.utils.BlockNumber, random.Random, raiden.transfer.state.NettingChannelState]) -> Union[TransitionResult, int]:
    if type(state_change) == Block:
        assert isinstance(state_change, Block), MYPY_ANNOTATION
        iteration = handle_block(initiator_state, state_change, channel_state, pseudo_random_generator)
    elif type(state_change) == ReceiveSecretRequest:
        assert isinstance(state_change, ReceiveSecretRequest), MYPY_ANNOTATION
        iteration = handle_secretrequest(initiator_state, state_change, channel_state, pseudo_random_generator)
    elif type(state_change) == ReceiveSecretReveal:
        assert isinstance(state_change, ReceiveSecretReveal), MYPY_ANNOTATION
        iteration = handle_offchain_secretreveal(initiator_state, state_change, channel_state, pseudo_random_generator, block_number)
    elif type(state_change) == ContractReceiveSecretReveal:
        assert isinstance(state_change, ContractReceiveSecretReveal), MYPY_ANNOTATION
        iteration = handle_onchain_secretreveal(initiator_state, state_change, channel_state, pseudo_random_generator, block_number)
    else:
        iteration = TransitionResult(initiator_state, [])
    return iteration