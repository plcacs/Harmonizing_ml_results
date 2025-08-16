from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with, run_epoch_processing_to
from eth2spec.test.context import spec_state_test, with_electra_and_later
from eth2spec.test.helpers.state import next_epoch_with_full_participation
from eth2spec.test.helpers.withdrawals import set_eth1_withdrawal_credential_with_balance, set_compounding_withdrawal_credential_with_balance
from eth2spec.test.helpers.typing import Spec, State

@with_electra_and_later
@spec_state_test
def test_basic_pending_consolidation(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_consolidation_not_yet_withdrawable_validator(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_skip_consolidation_when_source_slashed(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_all_consolidation_cases_together(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_future_epoch(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_compounding_creds(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_with_pending_deposit(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_source_balance_less_than_max_effective(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_source_balance_greater_than_max_effective(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_source_balance_less_than_max_effective_compounding(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_source_balance_greater_than_max_effective_compounding(spec: Spec, state: State):
    ...

def prepare_consolidation_and_state(spec: Spec, state: State, source_index: int, target_index: int, creds_type: str, balance_to_eb: str, eb_to_min_ab: str, eb_to_max_eb: str):
    ...

def run_balance_computation_test(spec: Spec, state: State, instance_tuples: list):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_balance_computation_eth1(spec: Spec, state: State):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_consolidation_balance_computation_compounding(spec: Spec, state: State):
    ...
