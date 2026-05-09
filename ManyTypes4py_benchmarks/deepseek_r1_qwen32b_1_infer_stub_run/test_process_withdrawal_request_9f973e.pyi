import random
from eth2spec.test.context import spec_state_test, with_electra_and_later, with_presets
from eth2spec.test.helpers.constants import MINIMAL
from eth2spec.spec.electra.types import Spec, BeaconState, WithdrawalRequest, PendingPartialWithdrawal
from typing import Generator, Tuple, List, Optional

def run_withdrawal_request_processing(spec: Spec, state: BeaconState, withdrawal_request: WithdrawalRequest, valid: bool = True, success: bool = True) -> Generator[Tuple[str, BeaconState | WithdrawalRequest | None], None, None]:
    ...

def compute_amount_to_withdraw(spec: Spec, state: BeaconState, index: int, amount: int) -> int:
    ...

@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request_with_first_validator(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request_with_compounding_credentials(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], 'need full partial withdrawal queue')
def test_basic_withdrawal_request_with_full_partial_withdrawal_queue(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_incorrect_source_address(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_incorrect_withdrawal_credential_prefix(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_on_withdrawal_request_initiated_exit_validator(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_activation_epoch_less_than_shard_committee_period(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_unknown_pubkey(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request_higher_excess_balance(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request_lower_than_excess_balance(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_pending_withdrawals(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_pending_withdrawals_and_high_amount(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_high_balance(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_high_amount(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_low_amount(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], 'need full partial withdrawal queue')
def test_partial_withdrawal_queue_full(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_no_compounding_credentials(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_no_excess_balance(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_pending_withdrawals_consume_all_excess_balance(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_insufficient_effective_balance(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_incorrect_source_address(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_incorrect_withdrawal_credential_prefix(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_on_exit_initiated_validator(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_activation_epoch_less_than_shard_committee_period(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_insufficient_balance(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_full_exit_request_has_partial_withdrawal(spec: Spec, state: BeaconState) -> None:
    ...

@with_electra_and_later
@spec_state_test
def test_incorrect_inactive_validator(spec: Spec, state: BeaconState) -> None:
    ...