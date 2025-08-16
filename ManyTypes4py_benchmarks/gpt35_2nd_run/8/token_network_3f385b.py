from raiden.transfer.state import TokenNetworkState
from raiden.transfer.state_change import ActionChannelClose, ActionChannelCoopSettle, ActionChannelSetRevealTimeout, ActionChannelWithdraw, ContractReceiveChannelBatchUnlock, ContractReceiveChannelClosed, ContractReceiveChannelDeposit, ContractReceiveChannelNew, ContractReceiveChannelSettled, ContractReceiveChannelWithdraw, ContractReceiveUpdateTransfer, ReceiveWithdrawConfirmation, ReceiveWithdrawExpired, ReceiveWithdrawRequest
from raiden.utils.typing import MYPY_ANNOTATION, BlockHash, BlockNumber, List, Union

StateChangeWithChannelID = Union[ActionChannelClose, ActionChannelCoopSettle, ActionChannelWithdraw, ActionChannelSetRevealTimeout, ContractReceiveChannelClosed, ContractReceiveChannelDeposit, ContractReceiveChannelSettled, ContractReceiveUpdateTransfer, ContractReceiveChannelWithdraw, ReceiveWithdrawConfirmation, ReceiveWithdrawRequest, ReceiveWithdrawExpired]

def subdispatch_to_channel_by_id(token_network_state: TokenNetworkState, state_change: StateChangeWithChannelID, block_number: BlockNumber, block_hash: BlockHash, pseudo_random_generator) -> TransitionResult:
    events: List[Event] = []
    ids_to_channels = token_network_state.channelidentifiers_to_channels
    channel_state = ids_to_channels.get(state_change.channel_identifier)
    if channel_state:
        result = channel.state_transition(channel_state=channel_state, state_change=state_change, block_number=block_number, block_hash=block_hash, pseudo_random_generator=pseudo_random_generator)
        partner_to_channelids = token_network_state.partneraddresses_to_channelidentifiers[channel_state.partner_state.address]
        channel_identifier = state_change.channel_identifier
        if result.new_state is None:
            del ids_to_channels[channel_identifier]
            partner_to_channelids.remove(channel_identifier)
        else:
            ids_to_channels[channel_identifier] = result.new_state
        events.extend(result.events)
    return TransitionResult(token_network_state, events)

def handle_channel_close(token_network_state: TokenNetworkState, state_change: ActionChannelClose, block_number: BlockNumber, block_hash: BlockHash, pseudo_random_generator) -> TransitionResult:
    return subdispatch_to_channel_by_id(token_network_state=token_network_state, state_change=state_change, block_number=block_number, block_hash=block_hash, pseudo_random_generator=pseudo_random_generator)

# Define other handle functions with appropriate type annotations
