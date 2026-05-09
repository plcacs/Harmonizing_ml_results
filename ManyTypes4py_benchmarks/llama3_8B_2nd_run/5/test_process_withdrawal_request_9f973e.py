from typing import Callable, List, Tuple

import random
from eth2spec.test.context import spec_state_test, expect_assertion_error, with_electra_and_later, with_presets
from eth2spec.test.helpers.constants import MINIMAL
from eth2spec.test.helpers.state import get_validator_index_by_pubkey
from eth2spec.test.helpers.withdrawals import set_eth1_withdrawal_credential_with_balance, set_compounding_withdrawal_credential

def run_withdrawal_request_processing(
    spec: Callable, 
    state: Tuple, 
    withdrawal_request: object, 
    valid: bool = True, 
    success: bool = True
) -> List[Tuple]:
    ...

@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_request(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_no_compounding_credentials(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_no_excess_balance(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_pending_withdrawals_consume_all_excess_balance(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_insufficient_effective_balance(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_incorrect_source_address(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_incorrect_withdrawal_credential_prefix(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_on_exit_initiated_validator(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_activation_epoch_less_than_shard_committee_period(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_insufficient_balance(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_full_exit_request_has_partial_withdrawal(spec: Callable, state: Tuple):
    ...

@with_electra_and_later
@spec_state_test
def test_incorrect_inactive_validator(spec: Callable, state: Tuple):
    ...
