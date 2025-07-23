from typing import Generator, List, Tuple, Any
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with, run_epoch_processing_to
from eth2spec.test.context import spec_state_test, with_electra_and_later
from eth2spec.test.helpers.state import next_epoch_with_full_participation
from eth2spec.test.helpers.withdrawals import set_eth1_withdrawal_credential_with_balance, set_compounding_withdrawal_credential_with_balance

@with_electra_and_later
@spec_state_test
def test_basic_pending_consolidation(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    state.validators[source_index].withdrawable_epoch = current_epoch
    eth1_withdrawal_credential = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target_index].withdrawal_credentials = eth1_withdrawal_credential
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[target_index] == 2 * spec.MIN_ACTIVATION_BALANCE
    assert state.balances[source_index] == 0
    assert state.pending_consolidations == []

@with_electra_and_later
@spec_state_test
def test_consolidation_not_yet_withdrawable_validator(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    eth1_withdrawal_credential = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target_index].withdrawal_credentials = eth1_withdrawal_credential
    spec.initiate_validator_exit(state, source_index)
    pre_pending_consolidations = state.pending_consolidations.copy()
    pre_balances = state.balances.copy()
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[source_index] == pre_balances[0]
    assert state.balances[target_index] == pre_balances[1]
    assert state.pending_consolidations == pre_pending_consolidations

@with_electra_and_later
@spec_state_test
def test_skip_consolidation_when_source_slashed(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source0_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target0_index = spec.get_active_validator_indices(state, current_epoch)[1]
    source1_index = spec.get_active_validator_indices(state, current_epoch)[2]
    target1_index = spec.get_active_validator_indices(state, current_epoch)[3]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source0_index, target_index=target0_index))
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source1_index, target_index=target1_index))
    eth1_withdrawal_credential = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target0_index].withdrawal_credentials = eth1_withdrawal_credential
    state.validators[target1_index].withdrawal_credentials = eth1_withdrawal_credential
    state.validators[source0_index].withdrawable_epoch = spec.get_current_epoch(state)
    state.validators[source1_index].withdrawable_epoch = spec.get_current_epoch(state)
    state.validators[source0_index].slashed = True
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[target0_index] == spec.MIN_ACTIVATION_BALANCE
    assert state.balances[source0_index] == spec.MIN_ACTIVATION_BALANCE
    assert state.balances[target1_index] == 2 * spec.MIN_ACTIVATION_BALANCE
    assert state.balances[source1_index] == 0

@with_electra_and_later
@spec_state_test
def test_all_consolidation_cases_together(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = [spec.get_active_validator_indices(state, current_epoch)[i] for i in range(4)]
    target_index = [spec.get_active_validator_indices(state, current_epoch)[4 + i] for i in range(4)]
    state.pending_consolidations = [spec.PendingConsolidation(source_index=source_index[i], target_index=target_index[i]) for i in range(4)]
    for i in [0, 2]:
        state.validators[source_index[i]].withdrawable_epoch = current_epoch
    state.validators[source_index[1]].slashed = True
    eth1_withdrawal_credential = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    for i in range(4):
        state.validators[target_index[i]].withdrawal_credentials = eth1_withdrawal_credential
    spec.initiate_validator_exit(state, 2)
    pre_balances = state.balances.copy()
    pre_pending_consolidations = state.pending_consolidations.copy()
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[target_index[0]] == 2 * spec.MIN_ACTIVATION_BALANCE
    assert state.balances[source_index[0]] == 0
    for i in [1, 2, 3]:
        assert state.balances[source_index[i]] == pre_balances[source_index[i]]
        assert state.balances[target_index[i]] == pre_balances[target_index[i]]
    state.pending_consolidations = pre_pending_consolidations[2:]

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_future_epoch(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    spec.initiate_validator_exit(state, source_index)
    state.validators[source_index].withdrawable_epoch = state.validators[source_index].exit_epoch + spec.Epoch(1)
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    eth1_withdrawal_credential = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target_index].withdrawal_credentials = eth1_withdrawal_credential
    target_epoch = state.validators[source_index].withdrawable_epoch - spec.Epoch(1)
    while spec.get_current_epoch(state) < target_epoch:
        next_epoch_with_full_participation(spec, state)
    state_before_consolidation = state.copy()
    run_epoch_processing_to(spec, state_before_consolidation, 'process_pending_consolidations')
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    expected_source_balance = state_before_consolidation.balances[source_index] - spec.MIN_ACTIVATION_BALANCE
    expected_target_balance = state_before_consolidation.balances[target_index] + spec.MIN_ACTIVATION_BALANCE
    assert state.balances[source_index] == expected_source_balance
    assert state.balances[target_index] == expected_target_balance
    assert state.pending_consolidations == []

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_compounding_creds(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    spec.initiate_validator_exit(state, source_index)
    state.validators[source_index].withdrawable_epoch = state.validators[source_index].exit_epoch + spec.Epoch(1)
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    state.validators[source_index].withdrawal_credentials = spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target_index].withdrawal_credentials = spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x12' * 20
    target_epoch = state.validators[source_index].withdrawable_epoch - spec.Epoch(1)
    while spec.get_current_epoch(state) < target_epoch:
        next_epoch_with_full_participation(spec, state)
    state_before_consolidation = state.copy()
    run_epoch_processing_to(spec, state_before_consolidation, 'process_pending_consolidations')
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    expected_target_balance = spec.MIN_ACTIVATION_BALANCE + state_before_consolidation.balances[target_index]
    assert state.balances[target_index] == expected_target_balance
    assert state.balances[source_index] == state_before_consolidation.balances[source_index] - spec.MIN_ACTIVATION_BALANCE
    assert state.pending_consolidations == []
    assert len(state.pending_deposits) == 0

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_with_pending_deposit(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    spec.initiate_validator_exit(state, source_index)
    source = state.validators[source_index]
    source.withdrawable_epoch = source.exit_epoch + spec.Epoch(1)
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    pending_deposit = spec.PendingDeposit(pubkey=source.pubkey, withdrawal_credentials=source.withdrawal_credentials, amount=spec.MIN_ACTIVATION_BALANCE, signature=spec.bls.G2_POINT_AT_INFINITY, slot=spec.GENESIS_SLOT)
    state.pending_deposits.append(pending_deposit)
    state.validators[source_index].withdrawal_credentials = spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x11' * 20
    state.validators[target_index].withdrawal_credentials = spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + b'\x12' * 20
    target_epoch = source.withdrawable_epoch - spec.Epoch(1)
    while spec.get_current_epoch(state) < target_epoch:
        next_epoch_with_full_participation(spec, state)
    state_before_consolidation = state.copy()
    run_epoch_processing_to(spec, state_before_consolidation, 'process_pending_consolidations')
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    expected_target_balance = spec.MIN_ACTIVATION_BALANCE + state_before_consolidation.balances[target_index]
    assert state.balances[target_index] == expected_target_balance
    assert state.balances[source_index] == state_before_consolidation.balances[source_index] - spec.MIN_ACTIVATION_BALANCE
    assert state.pending_consolidations == []
    assert state.pending_deposits == [pending_deposit]

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_source_balance_less_than_max_effective(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    state.validators[source_index].withdrawable_epoch = current_epoch
    set_eth1_withdrawal_credential_with_balance(spec, state, source_index)
    set_eth1_withdrawal_credential_with_balance(spec, state, target_index)
    pre_balance_source = state.validators[source_index].effective_balance - spec.EFFECTIVE_BALANCE_INCREMENT // 8
    state.balances[source_index] = pre_balance_source
    pre_balance_target = state.balances[target_index]
    assert state.balances[source_index] < spec.get_max_effective_balance(state.validators[source_index])
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[target_index] == pre_balance_target + pre_balance_source
    assert state.balances[source_index] == 0
    assert state.pending_consolidations == []

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_source_balance_greater_than_max_effective(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    state.validators[source_index].withdrawable_epoch = current_epoch
    set_eth1_withdrawal_credential_with_balance(spec, state, source_index)
    set_eth1_withdrawal_credential_with_balance(spec, state, target_index)
    excess_source_balance = spec.EFFECTIVE_BALANCE_INCREMENT // 8
    pre_balance_source = state.validators[source_index].effective_balance + excess_source_balance
    state.balances[source_index] = pre_balance_source
    pre_balance_target = state.balances[target_index]
    source_max_effective_balance = spec.get_max_effective_balance(state.validators[source_index])
    assert state.balances[source_index] > source_max_effective_balance
    yield from run_epoch_processing_with(spec, state, 'process_pending_consolidations')
    assert state.balances[target_index] == pre_balance_target + source_max_effective_balance
    assert state.balances[source_index] == excess_source_balance
    assert state.pending_consolidations == []

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_source_balance_less_than_max_effective_compounding(spec: Any, state: Any) -> Generator[None, None, None]:
    current_epoch = spec.get_current_epoch(state)
    source_index = spec.get_active_validator_indices(state, current_epoch)[0]
    target_index = spec.get_active_validator_indices(state, current_epoch)[1]
    state.pending_consolidations.append(spec.PendingConsolidation(source_index=source_index, target_index=target_index))
    state.validators[source_index].withdrawable_epoch = current_epoch
    set_compounding_withdrawal_credential_with_balance(spec, state, source_index)
    set_compounding_withdrawal_credential_with_balance(spec, state, target_index)
    pre_balance_source = state.validators[source_index].effective_balance - spec.EFFECTIVE_BALANCE_INCREMENT // 8
    state.balances[source_index] = pre_balance_source
    pre_balance_target = state.balances[target_index]
    assert state.balances[source_index] < spec.get_max_effective_balance(state.validators[source_index])
    yield from run_epoch_