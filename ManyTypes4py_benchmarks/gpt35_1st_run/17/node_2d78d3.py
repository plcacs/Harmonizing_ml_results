from raiden.transfer.state import ChainState

def get_token_network_by_address(chain_state: ChainState, token_network_address: TokenNetworkAddress) -> Optional[TokenNetworkState]:
def subdispatch_to_all_channels(chain_state: ChainState, state_change: TokenNetworkStateChange, block_number: BlockNumber, block_hash: BlockHash) -> TransitionResult:
def subdispatch_by_canonical_id(chain_state: ChainState, state_change: TokenNetworkStateChange, canonical_identifier: CanonicalIdentifier) -> TransitionResult:
def subdispatch_to_all_lockedtransfers(chain_state: ChainState, state_change: TokenNetworkStateChange) -> TransitionResult:
def subdispatch_to_paymenttask(chain_state: ChainState, state_change: TokenNetworkStateChange, secrethash: SecretHash) -> TransitionResult:
def subdispatch_initiatortask(chain_state: ChainState, state_change: TokenNetworkStateChange, token_network_address: TokenNetworkAddress, secrethash: SecretHash) -> TransitionResult:
def subdispatch_mediatortask(chain_state: ChainState, state_change: TokenNetworkStateChange, token_network_address: TokenNetworkAddress, secrethash: SecretHash) -> TransitionResult:
def subdispatch_targettask(chain_state: ChainState, state_change: TokenNetworkStateChange, token_network_address: TokenNetworkAddress, channel_identifier: ChannelID, secrethash: SecretHash) -> TransitionResult:
def maybe_add_tokennetwork(chain_state: ChainState, token_network_registry_address: TokenNetworkRegistryAddress, token_network_state: TokenNetworkState) -> None:
def inplace_delete_message_queue(chain_state: ChainState, state_change: StateChange, queueid: QueueIdentifier) -> None:
def inplace_delete_message(message_queue: List[Event], state_change: StateChange) -> None:
def handle_block(chain_state: ChainState, state_change: Block) -> TransitionResult:
def handle_token_network_action(chain_state: ChainState, state_change: TokenNetworkStateChange) -> TransitionResult:
def handle_contract_receive_channel_closed(chain_state: ChainState, state_change: ContractReceiveChannelClosed) -> TransitionResult:
def handle_receive_delivered(chain_state: ChainState, state_change: ReceiveDelivered) -> TransitionResult:
def handle_contract_receive_new_token_network_registry(chain_state: ChainState, state_change: ContractReceiveNewTokenNetworkRegistry) -> TransitionResult:
def handle_contract_receive_new_token_network(chain_state: ChainState, state_change: ContractReceiveNewTokenNetwork) -> TransitionResult:
def handle_receive_secret_reveal(chain_state: ChainState, state_change: ReceiveSecretReveal) -> TransitionResult:
def handle_contract_receive_secret_reveal(chain_state: ChainState, state_change: ContractReceiveSecretReveal) -> TransitionResult:
def handle_action_init_initiator(chain_state: ChainState, state_change: ActionInitInitiator) -> TransitionResult:
def handle_action_init_mediator(chain_state: ChainState, state_change: ActionInitMediator) -> TransitionResult:
def handle_action_init_target(chain_state: ChainState, state_change: ActionInitTarget) -> TransitionResult:
def handle_action_transfer_reroute(chain_state: ChainState, state_change: ActionTransferReroute) -> TransitionResult:
def handle_receive_withdraw_request(chain_state: ChainState, state_change: ReceiveWithdrawRequest) -> TransitionResult:
def handle_receive_withdraw_confirmation(chain_state: ChainState, state_change: ReceiveWithdrawConfirmation) -> TransitionResult:
def handle_receive_withdraw_expired(chain_state: ChainState, state_change: ReceiveWithdrawExpired) -> TransitionResult:
def handle_receive_lock_expired(chain_state: ChainState, state_change: ReceiveLockExpired) -> TransitionResult:
def handle_receive_transfer_refund(chain_state: ChainState, state_change: ReceiveTransferRefund) -> TransitionResult:
def handle_receive_transfer_cancel_route(chain_state: ChainState, state_change: ReceiveTransferCancelRoute) -> TransitionResult:
def handle_receive_secret_request(chain_state: ChainState, state_change: ReceiveSecretRequest) -> TransitionResult:
def handle_receive_processed(chain_state: ChainState, state_change: ReceiveProcessed) -> TransitionResult:
def handle_update_services_addresses_state_change(chain_state: ChainState, state_change: UpdateServicesAddressesStateChange) -> TransitionResult:
def handle_receive_unlock(chain_state: ChainState, state_change: ReceiveUnlock) -> TransitionResult:
def handle_state_change(chain_state: ChainState, state_change: StateChange) -> TransitionResult:
def is_transaction_effect_satisfied(chain_state: ChainState, transaction: ContractSendEvent, state_change: ContractReceiveStateChange) -> bool:
def is_transaction_invalidated(transaction: ContractSendEvent, state_change: ContractReceiveStateChange) -> bool:
def is_transaction_expired(transaction: ContractSendEvent, block_number: BlockNumber) -> bool:
def is_transaction_pending(chain_state: ChainState, transaction: ContractSendEvent, state_change: ContractReceiveStateChange) -> bool:
def update_queues(iteration: TransitionResult, state_change: StateChange) -> None:
def state_transition(chain_state: ChainState, state_change: StateChange) -> TransitionResult:
