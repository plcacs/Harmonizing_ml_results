from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with, run_epoch_processing_to
from eth2spec.test.context import spec_state_test, with_electra_and_later
from eth2spec.test.helpers.state import next_epoch_with_full_participation
from eth2spec.test.helpers.withdrawals import set_eth1_withdrawal_credential_with_balance, set_compounding_withdrawal_credential_with_balance
from eth2spec import spec

@with_electra_and_later
@spec_state_test
def test_basic_pending_consolidation(spec: spec.Spec, state: spec.BeaconState) -> None:
    current_epoch: spec.Epoch = spec.get_current_epoch(state)
    source_index: spec.ValidatorIndex = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index: spec.ValidatorIndex = spec.get_active_validator_indices(state, current_epoch)[1]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    state.validators[source_index].withdrawable_epoch = current_epoch
    eth1_withdrawal_credential: bytes = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target_index].withdrawal_credentials = eth1_withdrawal_credential
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[target_index] == 2 * spec.MIN_ACTIVATION_BALANCE
    assert state.balances[source_index] == 0
    assert state.pending_consolidations == []

@with_electra_and_later
@spec_state_test
def test_consolidation_not_yet_withdrawable_validator(spec: spec.Spec, state: spec.BeaconState) -> None:
    current_epoch: spec.Epoch = spec.get_current_epoch(state)
    source_index: spec.ValidatorIndex = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index: spec.ValidatorIndex = spec.get_active_validator_indices(state, current_epoch)[1]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    eth1_withdrawal_credential: bytes = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target_index].withdrawal_credentials = eth1_withdrawal_credential
    spec.initiate_validator_exit(state, source_index)
    pre_pending_consolidations: List[spec.PendingConsolidation] = state.pending_consolidations.copy()
    pre_balances: List[spec.Gwei] = state.balances.copy()
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[source_index] == pre_balances[0]
    assert state.balances[target_index] == pre_balances[1]
    assert state.pending_consolidations == pre_pending_consolidations

# Add annotations for remaining functions in a similar manner
