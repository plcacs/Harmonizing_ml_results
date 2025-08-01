import random
from typing import Optional, List

from raiden.transfer import channel, secret_registry
from raiden.transfer.architecture import Event, StateChange, TransitionResult
from raiden.transfer.events import EventPaymentReceivedSuccess
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE
from raiden.transfer.mediated_transfer.events import (
    EventUnlockClaimFailed,
    EventUnlockClaimSuccess,
    SendSecretRequest,
    SendSecretReveal,
)
from raiden.transfer.mediated_transfer.mediator import is_safe_to_wait
from raiden.transfer.mediated_transfer.state import TargetTransferState
from raiden.transfer.mediated_transfer.state_change import (
    ActionInitTarget,
    ReceiveLockExpired,
    ReceiveSecretReveal,
)
from raiden.transfer.state import NettingChannelState, message_identifier_from_prng
from raiden.transfer.state_change import (
    Block,
    ContractReceiveSecretReveal,
    ReceiveUnlock,
)
from raiden.transfer.utils import is_valid_secret_reveal
from raiden.utils.typing import (
    Address,
    BlockHash,
    BlockNumber,
    PaymentAmount,
)

def sanity_check(
    old_state: Optional[TargetTransferState],
    new_state: Optional[TargetTransferState],
    channel_state: NettingChannelState,
) -> None:
    was_running = old_state is not None
    is_running = new_state is not None
    is_cleared = was_running and (not is_running)
    if old_state and is_cleared:
        lock = channel.get_lock(
            end_state=channel_state.partner_state,
            secrethash=old_state.transfer.lock.secrethash,
        )
        assert lock is None, "The lock must be cleared once the task exists"
    elif new_state and is_running:
        lock = channel.get_lock(
            end_state=channel_state.partner_state,
            secrethash=new_state.transfer.lock.secrethash,
        )
        assert lock is not None, "The lock must not be cleared while the task is running"

def events_for_onchain_secretreveal(
    target_state: TargetTransferState,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> List[Event]:
    """Emits the event for revealing the secret on-chain if the transfer
    can not be settled off-chain.
    """
    transfer = target_state.transfer
    expiration = transfer.lock.expiration
    safe_to_wait = is_safe_to_wait(
        expiration, channel_state.reveal_timeout, block_number
    )
    secret_known_offchain = channel.is_secret_known_offchain(
        channel_state.partner_state, transfer.lock.secrethash
    )
    has_onchain_reveal_started = target_state.state == TargetTransferState.ONCHAIN_SECRET_REVEAL
    if not safe_to_wait and secret_known_offchain and (not has_onchain_reveal_started):
        target_state.state = TargetTransferState.ONCHAIN_SECRET_REVEAL
        secret = channel.get_secret(
            channel_state.partner_state, transfer.lock.secrethash
        )
        assert secret, "secret should be known at this point"
        return secret_registry.events_for_onchain_secretreveal(
            channel_state=channel_state,
            secret=secret,
            expiration=expiration,
            block_hash=block_hash,
        )
    return []

def handle_inittarget(
    state_change: ActionInitTarget,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState], List[Event]]:
    """Handles an ActionInitTarget state change."""
    transfer = state_change.transfer
    from_hop = state_change.from_hop
    assert (
        channel_state.identifier == transfer.balance_proof.channel_identifier
    ), "channel_id mismatch in handle_inittarget"
    is_valid, channel_events, errormsg = channel.handle_receive_lockedtransfer(
        channel_state, transfer, transfer.payer_address_metadata
    )
    if is_valid:
        target_state = TargetTransferState(from_hop, transfer)
        if state_change.received_valid_secret:
            return TransitionResult(target_state, channel_events)
        safe_to_wait = is_safe_to_wait(
            transfer.lock.expiration, channel_state.reveal_timeout, block_number
        )
        if safe_to_wait:
            message_identifier = message_identifier_from_prng(
                pseudo_random_generator
            )
            recipient = transfer.initiator
            secret_request = SendSecretRequest(
                recipient=Address(recipient),
                recipient_metadata=transfer.initiator_address_metadata,
                message_identifier=message_identifier,
                payment_identifier=transfer.payment_identifier,
                amount=PaymentAmount(transfer.lock.amount),
                expiration=transfer.lock.expiration,
                secrethash=transfer.lock.secrethash,
                canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE,
            )
            channel_events.append(secret_request)
        iteration = TransitionResult(target_state, channel_events)
    else:
        assert errormsg, "handle_receive_lockedtransfer should return error msg if not valid"
        unlock_failed = EventUnlockClaimFailed(
            identifier=transfer.payment_identifier,
            secrethash=transfer.lock.secrethash,
            reason=errormsg,
        )
        channel_events.append(unlock_failed)
        iteration = TransitionResult(None, channel_events)
    return iteration

def handle_offchain_secretreveal(
    target_state: TargetTransferState,
    state_change: ReceiveSecretReveal,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState], List[Event]]:
    """Validates and handles a ReceiveSecretReveal state change."""
    valid_secret = is_valid_secret_reveal(
        state_change=state_change,
        transfer_secrethash=target_state.transfer.lock.secrethash,
    )
    has_transfer_expired = channel.is_transfer_expired(
        transfer=target_state.transfer,
        affected_channel=channel_state,
        block_number=block_number,
    )
    if valid_secret and (not has_transfer_expired):
        channel.register_offchain_secret(
            channel_state=channel_state,
            secret=state_change.secret,
            secrethash=state_change.secrethash,
        )
        from_hop = target_state.from_hop
        message_identifier = message_identifier_from_prng(pseudo_random_generator)
        target_state.state = TargetTransferState.OFFCHAIN_SECRET_REVEAL
        target_state.secret = state_change.secret
        recipient = from_hop.node_address
        reveal = SendSecretReveal(
            recipient=recipient,
            recipient_metadata=target_state.transfer.payer_address_metadata,
            message_identifier=message_identifier,
            canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE,
            secret=target_state.secret,
        )
        iteration = TransitionResult(target_state, [reveal])
    else:
        iteration = TransitionResult(target_state, [])
    return iteration

def handle_onchain_secretreveal(
    target_state: TargetTransferState,
    state_change: ContractReceiveSecretReveal,
    channel_state: NettingChannelState,
) -> TransitionResult[TargetTransferState, List[Event]]:
    """Validates and handles a ContractReceiveSecretReveal state change."""
    valid_secret = is_valid_secret_reveal(
        state_change=state_change,
        transfer_secrethash=target_state.transfer.lock.secrethash,
    )
    if valid_secret:
        channel.register_onchain_secret(
            channel_state=channel_state,
            secret=state_change.secret,
            secrethash=state_change.secrethash,
            secret_reveal_block_number=state_change.block_number,
        )
        target_state.state = TargetTransferState.ONCHAIN_UNLOCK
        target_state.secret = state_change.secret
    return TransitionResult(target_state, [])

def handle_unlock(
    target_state: TargetTransferState,
    state_change: ReceiveUnlock,
    channel_state: NettingChannelState,
) -> TransitionResult[Optional[TargetTransferState], List[Event]]:
    """Handles a ReceiveUnlock state change."""
    recipient_metadata = target_state.transfer.payer_address_metadata
    is_valid, events, _ = channel.handle_unlock(
        channel_state, state_change, recipient_metadata
    )
    next_target_state: Optional[TargetTransferState] = target_state
    if is_valid:
        transfer = target_state.transfer
        payment_received_success = EventPaymentReceivedSuccess(
            token_network_registry_address=channel_state.token_network_registry_address,
            token_network_address=channel_state.token_network_address,
            identifier=transfer.payment_identifier,
            amount=PaymentAmount(transfer.lock.amount),
            initiator=transfer.initiator,
        )
        unlock_success = EventUnlockClaimSuccess(
            transfer.payment_identifier, transfer.lock.secrethash
        )
        events.extend([payment_received_success, unlock_success])
        next_target_state = None
    return TransitionResult(next_target_state, events)

def handle_block(
    target_state: TargetTransferState,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
    block_hash: BlockHash,
) -> TransitionResult[TargetTransferState, List[Event]]:
    """After Raiden learns about a new block this function must be called to
    handle expiration of the hash time lock.
    """
    transfer = target_state.transfer
    events: List[Event] = []
    lock = transfer.lock
    secret_known = channel.is_secret_known(
        channel_state.partner_state, lock.secrethash
    )
    lock_has_expired = channel.is_lock_expired(
        end_state=channel_state.our_state,
        lock=lock,
        block_number=block_number,
        lock_expiration_threshold=channel.get_receiver_expiration_threshold(
            lock.expiration
        ),
    )
    if lock_has_expired and target_state.state != TargetTransferState.EXPIRED:
        failed = EventUnlockClaimFailed(
            identifier=transfer.payment_identifier,
            secrethash=transfer.lock.secrethash,
            reason="lock expired",
        )
        target_state.state = TargetTransferState.EXPIRED
        events = [failed]
    elif secret_known:
        events = events_for_onchain_secretreveal(
            target_state=target_state,
            channel_state=channel_state,
            block_number=block_number,
            block_hash=block_hash,
        )
    return TransitionResult(target_state, events)

def handle_lock_expired(
    target_state: TargetTransferState,
    state_change: ReceiveLockExpired,
    channel_state: NettingChannelState,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState], List[Event]]:
    """Remove expired locks from channel states."""
    recipient_metadata = target_state.transfer.payer_address_metadata
    result = channel.handle_receive_lock_expired(
        channel_state=channel_state,
        state_change=state_change,
        block_number=block_number,
        recipient_metadata=recipient_metadata,
    )
    assert result.new_state, "handle_receive_lock_expired should not delete the task"
    if not channel.get_lock(
        result.new_state.partner_state, target_state.transfer.lock.secrethash
    ):
        transfer = target_state.transfer
        unlock_failed = EventUnlockClaimFailed(
            identifier=transfer.payment_identifier,
            secrethash=transfer.lock.secrethash,
            reason="Lock expired",
        )
        result.events.append(unlock_failed)
        return TransitionResult(None, result.events)
    return TransitionResult(target_state, result.events)

def state_transition(
    target_state: Optional[TargetTransferState],
    state_change: StateChange,
    channel_state: NettingChannelState,
    pseudo_random_generator: random.Random,
    block_number: BlockNumber,
) -> TransitionResult[Optional[TargetTransferState], List[Event]]:
    """State machine for the target node of a mediated transfer."""
    iteration: TransitionResult[
        Optional[TargetTransferState], List[Event]
    ] = TransitionResult(target_state, [])
    if isinstance(state_change, ActionInitTarget):
        if target_state is None:
            iteration = handle_inittarget(
                state_change, channel_state, pseudo_random_generator, block_number
            )
    elif isinstance(state_change, Block):
        assert (
            state_change.block_number == block_number
        ), "Block number mismatch"
        assert target_state, "Block state changes should be accompanied by a valid target state"
        iteration = handle_block(
            target_state=target_state,
            channel_state=channel_state,
            block_number=state_change.block_number,
            block_hash=state_change.block_hash,
        )
    elif isinstance(state_change, ReceiveSecretReveal):
        assert (
            target_state
        ), "ReceiveSecretReveal should be accompanied by a valid target state"
        iteration = handle_offchain_secretreveal(
            target_state=target_state,
            state_change=state_change,
            channel_state=channel_state,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number,
        )
    elif isinstance(state_change, ContractReceiveSecretReveal):
        msg = "ContractReceiveSecretReveal should be accompanied by a valid target state"
        assert target_state, msg
        iteration = handle_onchain_secretreveal(
            target_state, state_change, channel_state
        )
    elif isinstance(state_change, ReceiveUnlock):
        assert (
            target_state
        ), "ReceiveUnlock should be accompanied by a valid target state"
        iteration = handle_unlock(
            target_state=target_state,
            state_change=state_change,
            channel_state=channel_state,
        )
    elif isinstance(state_change, ReceiveLockExpired):
        assert (
            target_state
        ), "ReceiveLockExpired should be accompanied by a valid target state"
        iteration = handle_lock_expired(
            target_state=target_state,
            state_change=state_change,
            channel_state=channel_state,
            block_number=block_number,
        )
    sanity_check(
        old_state=target_state,
        new_state=iteration.new_state,
        channel_state=channel_state,
    )
    return iteration
