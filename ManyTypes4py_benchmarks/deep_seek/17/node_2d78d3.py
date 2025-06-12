from typing import Dict, List, Optional, Set, Tuple, Union, Any, TypeVar, Generic, cast
from raiden.transfer import channel, token_network, views
from raiden.transfer.architecture import ContractReceiveStateChange, ContractSendEvent, Event, SendMessageEvent, StateChange, TransitionResult
from raiden.transfer.events import ContractSendChannelBatchUnlock, ContractSendChannelClose, ContractSendChannelSettle, ContractSendChannelUpdateTransfer, ContractSendChannelWithdraw, ContractSendSecretReveal, SendWithdrawRequest, UpdateServicesAddresses
from raiden.transfer.identifiers import CANONICAL_IDENTIFIER_UNORDERED_QUEUE, CanonicalIdentifier, QueueIdentifier
from raiden.transfer.mediated_transfer import initiator_manager, mediator, target
from raiden.transfer.mediated_transfer.state import InitiatorPaymentState, MediatorTransferState, TargetTransferState
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ActionInitMediator, ActionInitTarget, ActionTransferReroute, ReceiveLockExpired, ReceiveSecretRequest, ReceiveSecretReveal, ReceiveTransferCancelRoute, ReceiveTransferRefund
from raiden.transfer.mediated_transfer.tasks import InitiatorTask, MediatorTask, TargetTask
from raiden.transfer.state import ChainState, TokenNetworkRegistryState, TokenNetworkState
from raiden.transfer.state_change import ActionChannelClose, ActionChannelCoopSettle, ActionChannelSetRevealTimeout, ActionChannelWithdraw, Block, ContractReceiveChannelBatchUnlock, ContractReceiveChannelClosed, ContractReceiveChannelDeposit, ContractReceiveChannelNew, ContractReceiveChannelSettled, ContractReceiveChannelWithdraw, ContractReceiveNewTokenNetwork, ContractReceiveNewTokenNetworkRegistry, ContractReceiveSecretReveal, ContractReceiveUpdateTransfer, ReceiveDelivered, ReceiveProcessed, ReceiveUnlock, ReceiveWithdrawConfirmation, ReceiveWithdrawExpired, ReceiveWithdrawRequest, UpdateServicesAddressesStateChange
from raiden.utils.copy import deepcopy
from raiden.utils.typing import MYPY_ANNOTATION, Any, BlockHash, BlockNumber, ChannelID, Dict, List, Optional, SecretHash, TokenNetworkAddress, TokenNetworkRegistryAddress, Union, typecheck

TokenNetworkStateChange = Union[
    ActionChannelClose,
    ContractReceiveChannelBatchUnlock,
    ContractReceiveChannelNew,
    ContractReceiveChannelDeposit,
    ContractReceiveChannelSettled,
    ContractReceiveUpdateTransfer,
    ContractReceiveChannelClosed,
    ContractReceiveChannelWithdraw
]

def get_token_network_by_address(
    chain_state: ChainState,
    token_network_address: TokenNetworkAddress
) -> Optional[TokenNetworkState]:
    tn_registry_address = chain_state.tokennetworkaddresses_to_tokennetworkregistryaddresses.get(token_network_address)
    if not tn_registry_address:
        return None
    tn_registry_state = chain_state.identifiers_to_tokennetworkregistries.get(tn_registry_address)
    if not tn_registry_state:
        return None
    return tn_registry_state.tokennetworkaddresses_to_tokennetworks.get(token_network_address)

def subdispatch_to_all_channels(
    chain_state: ChainState,
    state_change: StateChange,
    block_number: BlockNumber,
    block_hash: BlockHash
) -> TransitionResult:
    events: List[Event] = []
    for token_network_registry in chain_state.identifiers_to_tokennetworkregistries.values():
        for token_network_state in token_network_registry.tokennetworkaddresses_to_tokennetworks.values():
            for channel_state in token_network_state.channelidentifiers_to_channels.values():
                result = channel.state_transition(
                    channel_state=channel_state,
                    state_change=state_change,
                    block_number=block_number,
                    block_hash=block_hash,
                    pseudo_random_generator=chain_state.pseudo_random_generator
                )
                events.extend(result.events)
    return TransitionResult(chain_state, events)

def subdispatch_by_canonical_id(
    chain_state: ChainState,
    state_change: StateChange,
    canonical_identifier: CanonicalIdentifier
) -> TransitionResult:
    token_network_state = get_token_network_by_address(chain_state, canonical_identifier.token_network_address)
    events: List[Event] = []
    if token_network_state:
        iteration = token_network.state_transition(
            token_network_state=token_network_state,
            state_change=state_change,
            block_number=chain_state.block_number,
            block_hash=chain_state.block_hash,
            pseudo_random_generator=chain_state.pseudo_random_generator
        )
        assert iteration.new_state, 'No token network state transition can lead to None'
        events = iteration.events
    return TransitionResult(chain_state, events)

def subdispatch_to_all_lockedtransfers(
    chain_state: ChainState,
    state_change: StateChange
) -> TransitionResult:
    events: List[Event] = []
    for secrethash in list(chain_state.payment_mapping.secrethashes_to_task.keys()):
        result = subdispatch_to_paymenttask(chain_state, state_change, secrethash)
        events.extend(result.events)
    return TransitionResult(chain_state, events)

def subdispatch_to_paymenttask(
    chain_state: ChainState,
    state_change: StateChange,
    secrethash: SecretHash
) -> TransitionResult:
    block_number = chain_state.block_number
    block_hash = chain_state.block_hash
    sub_task = chain_state.payment_mapping.secrethashes_to_task.get(secrethash)
    events: List[Event] = []
    if sub_task:
        pseudo_random_generator = chain_state.pseudo_random_generator
        if isinstance(sub_task, InitiatorTask):
            token_network_address = sub_task.token_network_address
            token_network_state = get_token_network_by_address(chain_state, token_network_address)
            if token_network_state:
                channel_identifier_map = token_network_state.channelidentifiers_to_channels
                sub_iteration = initiator_manager.state_transition(
                    payment_state=sub_task.manager_state,
                    state_change=state_change,
                    channelidentifiers_to_channels=channel_identifier_map,
                    addresses_to_channel=chain_state.addresses_to_channel,
                    pseudo_random_generator=pseudo_random_generator,
                    block_number=block_number
                )
                events = sub_iteration.events
                if sub_iteration.new_state is None:
                    del chain_state.payment_mapping.secrethashes_to_task[secrethash]
        elif isinstance(sub_task, MediatorTask):
            token_network_address = sub_task.token_network_address
            token_network_state = get_token_network_by_address(chain_state, token_network_address)
            if token_network_state:
                channelids_to_channels = token_network_state.channelidentifiers_to_channels
                sub_iteration = mediator.state_transition(
                    mediator_state=sub_task.mediator_state,
                    state_change=state_change,
                    channelidentifiers_to_channels=channelids_to_channels,
                    addresses_to_channel=chain_state.addresses_to_channel,
                    pseudo_random_generator=pseudo_random_generator,
                    block_number=block_number,
                    block_hash=block_hash
                )
                events = sub_iteration.events
                if sub_iteration.new_state is None:
                    del chain_state.payment_mapping.secrethashes_to_task[secrethash]
        elif isinstance(sub_task, TargetTask):
            token_network_address = sub_task.token_network_address
            channel_identifier = sub_task.channel_identifier
            channel_state = views.get_channelstate_by_canonical_identifier(
                chain_state=chain_state,
                canonical_identifier=CanonicalIdentifier(
                    chain_identifier=chain_state.chain_id,
                    token_network_address=token_network_address,
                    channel_identifier=channel_identifier
                )
            )
            if channel_state:
                sub_iteration = target.state_transition(
                    target_state=sub_task.target_state,
                    state_change=state_change,
                    channel_state=channel_state,
                    pseudo_random_generator=pseudo_random_generator,
                    block_number=block_number
                )
                events = sub_iteration.events
                if sub_iteration.new_state is None:
                    del chain_state.payment_mapping.secrethashes_to_task[secrethash]
    return TransitionResult(chain_state, events)

def subdispatch_initiatortask(
    chain_state: ChainState,
    state_change: StateChange,
    token_network_address: TokenNetworkAddress,
    secrethash: SecretHash
) -> TransitionResult:
    token_network_state = get_token_network_by_address(chain_state, token_network_address)
    if not token_network_state:
        return TransitionResult(chain_state, [])
    sub_task = chain_state.payment_mapping.secrethashes_to_task.get(secrethash)
    if not sub_task:
        manager_state = None
    else:
        if not isinstance(sub_task, InitiatorTask) or token_network_address != sub_task.token_network_address:
            return TransitionResult(chain_state, [])
        manager_state = sub_task.manager_state
    iteration = initiator_manager.state_transition(
        payment_state=manager_state,
        state_change=state_change,
        channelidentifiers_to_channels=token_network_state.channelidentifiers_to_channels,
        addresses_to_channel=chain_state.addresses_to_channel,
        pseudo_random_generator=chain_state.pseudo_random_generator,
        block_number=chain_state.block_number
    )
    events = iteration.events
    if iteration.new_state:
        chain_state.payment_mapping.secrethashes_to_task[secrethash] = InitiatorTask(token_network_address, iteration.new_state)
    elif secrethash in chain_state.payment_mapping.secrethashes_to_task:
        del chain_state.payment_mapping.secrethashes_to_task[secrethash]
    return TransitionResult(chain_state, events)

def subdispatch_mediatortask(
    chain_state: ChainState,
    state_change: StateChange,
    token_network_address: TokenNetworkAddress,
    secrethash: SecretHash
) -> TransitionResult:
    block_number = chain_state.block_number
    block_hash = chain_state.block_hash
    sub_task = chain_state.payment_mapping.secrethashes_to_task.get(secrethash)
    if not sub_task:
        is_valid_subtask = True
        mediator_state = None
    elif sub_task and isinstance(sub_task, MediatorTask):
        is_valid_subtask = token_network_address == sub_task.token_network_address
        mediator_state = sub_task.mediator_state
    else:
        is_valid_subtask = False
    events: List[Event] = []
    if is_valid_subtask:
        token_network_state = get_token_network_by_address(chain_state, token_network_address)
        if token_network_state:
            pseudo_random_generator = chain_state.pseudo_random_generator
            iteration = mediator.state_transition(
                mediator_state=mediator_state,
                state_change=state_change,
                channelidentifiers_to_channels=token_network_state.channelidentifiers_to_channels,
                addresses_to_channel=chain_state.addresses_to_channel,
                pseudo_random_generator=pseudo_random_generator,
                block_number=block_number,
                block_hash=block_hash
            )
            events = iteration.events
            if iteration.new_state:
                sub_task = MediatorTask(token_network_address, iteration.new_state)
                if sub_task is not None:
                    chain_state.payment_mapping.secrethashes_to_task[secrethash] = sub_task
            elif secrethash in chain_state.payment_mapping.secrethashes_to_task:
                del chain_state.payment_mapping.secrethashes_to_task[secrethash]
    return TransitionResult(chain_state, events)

def subdispatch_targettask(
    chain_state: ChainState,
    state_change: StateChange,
    token_network_address: TokenNetworkAddress,
    channel_identifier: ChannelID,
    secrethash: SecretHash
) -> TransitionResult:
    block_number = chain_state.block_number
    sub_task = chain_state.payment_mapping.secrethashes_to_task.get(secrethash)
    if not sub_task:
        is_valid_subtask = True
        target_state = None
    elif sub_task and isinstance(sub_task, TargetTask):
        is_valid_subtask = token_network_address == sub_task.token_network_address
        target_state = sub_task.target_state
    else:
        is_valid_subtask = False
    events: List[Event] = []
    channel_state = None
    if is_valid_subtask:
        channel_state = views.get_channelstate_by_canonical_identifier(
            chain_state=chain_state,
            canonical_identifier=CanonicalIdentifier(
                chain_identifier=chain_state.chain_id,
                token_network_address=token_network_address,
                channel_identifier=channel_identifier
            )
        )
    if channel_state:
        pseudo_random_generator = chain_state.pseudo_random_generator
        iteration = target.state_transition(
            target_state=target_state,
            state_change=state_change,
            channel_state=channel_state,
            pseudo_random_generator=pseudo_random_generator,
            block_number=block_number
        )
        events = iteration.events
        if iteration.new_state:
            sub_task = TargetTask(
                canonical_identifier=channel_state.canonical_identifier,
                target_state=iteration.new_state
            )
            if sub_task is not None:
                chain_state.payment_mapping.secrethashes_to_task[secrethash] = sub_task
        elif secrethash in chain_state.payment_mapping.secrethashes_to_task:
            del chain_state.payment_mapping.secrethashes_to_task[secrethash]
    return TransitionResult(chain_state, events)

def maybe_add_tokennetwork(
    chain_state: ChainState,
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_network_state: TokenNetworkState
) -> None:
    token_network_address = token_network_state.address
    token_address = token_network_state.token_address
    token_network_registry_state, token_network_state_previous = views.get_networks(
        chain_state,
        token_network_registry_address,
        token_address
    )
    if token_network_registry_state is None:
        token_network_registry_state = TokenNetworkRegistryState(token_network_registry_address, [token_network_state])
        ids_to_payments = chain_state.identifiers_to_tokennetworkregistries
        ids_to_payments[token_network_registry_address] = token_network_registry_state
    if token_network_state_previous is None:
        token_network_registry_state.add_token_network(token_network_state)
        mapping = chain_state.tokennetworkaddresses_to_tokennetworkregistryaddresses
        mapping[token_network_address] = token_network_registry_address

def inplace_delete_message_queue(
    chain_state: ChainState,
    state_change: StateChange,
    queueid: QueueIdentifier
) -> None:
    """Filter messages from queue, if the queue becomes empty, cleanup the queue itself."""
    queue = chain_state.queueids_to_queues.get(queueid)
    if not queue:
        if queueid in chain_state.queueids_to_queues:
            chain_state.queueids_to_queues.pop(queueid)
        return
    inplace_delete_message(message_queue=queue, state_change=state_change)
    if len(queue) == 0:
        del chain_state.queueids_to_queues[queueid]
    else:
        chain_state.queueids_to_queues[queueid] = queue

def inplace_delete_message(
    message_queue: List[SendMessageEvent],
    state_change: StateChange
) -> None:
    """Check if the message exists in queue with ID `queueid` and exclude if found."""
    for message in list(message_queue):
        if isinstance(message, SendWithdrawRequest):
            if not isinstance(state_change, ReceiveWithdrawConfirmation):
                continue
        message_found = message.message_identifier == state_change.message_identifier and message.recipient == state_change.sender
        if message_found:
            message_queue.remove(message)

def handle_block(
    chain_state: ChainState,
    state_change: Block
) -> TransitionResult:
    block_number = state_change.block_number
    chain_state.block_number = block_number
    chain_state.block_hash = state_change.block_hash
    channels_result = subdispatch_to_all_channels(
        chain_state=chain_state,
        state_change=state_change,
        block_number=block_number,
        block_hash=chain_state.block_hash
    )
    transfers_result = subdispatch_to_all_lockedtransfers(chain_state, state_change)
    events = channels_result.events + transfers_result.events
    return TransitionResult(chain_state, events)

def handle_token_network_action(
    chain_state: ChainState,
    state_change: TokenNetworkStateChange
) -> TransitionResult:
    token_network_state = get_token_network_by_address(chain_state, state_change.token_network_address)
    events: List[Event] = []
    if token_network_state:
        iteration = token_network.state_transition(
            token_network_state=token_network_state,
            state_change=state_change,
            block_number=chain_state.block_number,
            block_hash=chain_state.block_hash,
            pseudo_random_generator=chain_state.pseudo_random_generator
        )
        assert iteration.new_state, 'No token network state transition leads to None'
        events = iteration.events
    return TransitionResult(chain_state, events)

def handle_contract_receive_channel_closed(
    chain_state: ChainState,
    state_change: ContractReceiveChannelClosed
) -> TransitionResult:
    canonical_identifier = CanonicalIdentifier(
        chain_identifier=chain_state.chain_id,
        token_network_address=state_change.token_network_address,
        channel_identifier=state_change.channel_identifier
    )
    channel_state = views.get_channelstate_by_canonical_identifier(
        chain_state=chain_state,
        canonical_identifier=canonical_identifier
    )
    if channel_state:
        queue_id = QueueIdentifier(
            recipient=channel_state.partner_state.address,
            canonical_identifier=canonical_identifier
        )
        if queue_id in chain_state.queueids_to_queues:
            chain_state.queueids_to_queues.pop(queue_id)
    return handle_token_network_action(chain_state=chain_state, state_change=state_change)

def handle_receive_delivered(
    chain_state: ChainState,
    state_change: ReceiveDelivered
) -> TransitionResult:
    """Check if the "Delivered" message exists in the global queue and delete if found."""
    queueid = QueueIdentifier(state_change.sender, CANONICAL_IDENTIFIER_UNORDER