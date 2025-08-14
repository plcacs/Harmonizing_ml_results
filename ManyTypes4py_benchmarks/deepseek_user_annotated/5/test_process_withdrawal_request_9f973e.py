import random
from typing import Any, Generator, Tuple, Optional, List
from eth2spec.test.context import (
    spec_state_test,
    expect_assertion_error,
    with_electra_and_later,
    with_presets,
)
from eth2spec.test.helpers.constants import MINIMAL
from eth2spec.test.helpers.state import (
    get_validator_index_by_pubkey,
)
from eth2spec.test.helpers.withdrawals import (
    set_eth1_withdrawal_credential_with_balance,
    set_compounding_withdrawal_credential,
)

def run_withdrawal_request_processing(
    spec: Any,
    state: Any,
    withdrawal_request: Any,
    valid: bool = True,
    success: bool = True
) -> Generator[Tuple[str, Optional[Any]], None, None]:
    """
    Run ``process_withdrawal_request``, yielding:
      - pre-state ('pre')
      - withdrawal_request ('withdrawal_request')
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    If ``success == False``, it doesn't initiate exit successfully
    """
    yield "pre", state
    yield "withdrawal_request", withdrawal_request

    if not valid:
        expect_assertion_error(
            lambda: spec.process_withdrawal_request(
                state, withdrawal_request
            )
        )
        yield "post", None
        return

    pre_state = state.copy()

    spec.process_withdrawal_request(
        state, withdrawal_request
    )

    yield "post", state

    if not success:
        assert pre_state == state
    else:
        validator_index = get_validator_index_by_pubkey(
            state, withdrawal_request.validator_pubkey
        )
        pre_exit_epoch = pre_state.validators[validator_index].exit_epoch
        pre_pending_partial_withdrawals = pre_state.pending_partial_withdrawals.copy()
        pre_balance = pre_state.balances[validator_index]
        pre_effective_balance = pre_state.validators[validator_index].effective_balance
        assert state.balances[validator_index] == pre_balance
        assert (
            state.validators[validator_index].effective_balance == pre_effective_balance
        )
        if withdrawal_request.amount == spec.FULL_EXIT_REQUEST_AMOUNT:
            assert pre_exit_epoch == spec.FAR_FUTURE_EPOCH
            assert state.validators[validator_index].exit_epoch < spec.FAR_FUTURE_EPOCH
            assert spec.get_pending_balance_to_withdraw(state, validator_index) == 0
            assert state.pending_partial_withdrawals == pre_pending_partial_withdrawals
        else:
            expected_amount_to_withdraw = compute_amount_to_withdraw(
                spec, pre_state, validator_index, withdrawal_request.amount
            )
            assert state.validators[validator_index].exit_epoch == spec.FAR_FUTURE_EPOCH
            expected_withdrawable_epoch = (
                state.earliest_exit_epoch
                + spec.config.MIN_VALIDATOR_WITHDRAWABILITY_DELAY
            )
            expected_partial_withdrawal = spec.PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=expected_amount_to_withdraw,
                withdrawable_epoch=expected_withdrawable_epoch,
            )
            assert (
                state.pending_partial_withdrawals
                == pre_pending_partial_withdrawals + [expected_partial_withdrawal]
            )


def compute_amount_to_withdraw(spec: Any, state: Any, index: int, amount: int) -> int:
    pending_balance_to_withdraw = spec.get_pending_balance_to_withdraw(state, index)
    return min(
        state.balances[index]
        - spec.MIN_ACTIVATION_BALANCE
        - pending_balance_to_withdraw,
        amount,
    )


@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1337)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH

    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request
    )


@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request_with_first_validator(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH

    current_epoch = spec.get_current_epoch(state)
    validator_index = spec.get_active_validator_indices(state, current_epoch)[0]
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request
    )


@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request_with_compounding_credentials(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1338)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH

    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request
    )


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], "need full partial withdrawal queue")
def test_basic_withdrawal_request_with_full_partial_withdrawal_queue(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1339)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    partial_withdrawal = spec.PendingPartialWithdrawal(
        validator_index=1, amount=1, withdrawable_epoch=current_epoch
    )
    state.pending_partial_withdrawals = [
        partial_withdrawal
    ] * spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT

    yield from run_withdrawal_request_processing(
        spec,
        state,
        withdrawal_request,
    )


@with_electra_and_later
@spec_state_test
def test_incorrect_source_address(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1340)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH

    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    incorrect_address = b"\x33" * 20
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    withdrawal_request = spec.WithdrawalRequest(
        source_address=incorrect_address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request, success=False
    )


@with_electra_and_later
@spec_state_test
def test_incorrect_withdrawal_credential_prefix(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1341)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH

    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    state.validators[validator_index].withdrawal_credentials = (
        spec.BLS_WITHDRAWAL_PREFIX
        + state.validators[validator_index].withdrawal_credentials[1:]
    )
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request, success=False
    )


@with_electra_and_later
@spec_state_test
def test_on_withdrawal_request_initiated_exit_validator(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1342)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH

    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    spec.initiate_validator_exit(state, validator_index)
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request, success=False
    )


@with_electra_and_later
@spec_state_test
def test_activation_epoch_less_than_shard_committee_period(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1343)
    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    assert spec.get_current_epoch(state) < (
        state.validators[validator_index].activation_epoch
        + spec.config.SHARD_COMMITTEE_PERIOD
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request, success=False
    )


@with_electra_and_later
@spec_state_test
def test_unknown_pubkey(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1344)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH

    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    address = b"\x22" * 20
    pubkey = spec.BLSPubkey(b"\x23" * 48)
    set_eth1_withdrawal_credential_with_balance(
        spec, state, validator_index, address=address
    )
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT,
    )

    yield from run_withdrawal_request_processing(
        spec, state, withdrawal_request, success=False
    )


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1344)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    amount = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += amount

    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount,
    )

    yield from run_withdrawal_request_processing(
        spec,
        state,
        withdrawal_request,
    )

    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(
        current_epoch
    )


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request_higher_excess_balance(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1345)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    amount = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount

    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount,
    )

    yield from run_withdrawal_request_processing(
        spec,
        state,
        withdrawal_request,
    )

    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(
        current_epoch
    )


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request_lower_than_excess_balance(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    rng = random.Random(1346)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch = spec.get_current_epoch(state)
    validator_index = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey = state.validators[validator_index].pubkey
    address = b"\x22" * 20
    excess_balance = spec.EFFECTIVE_BALANCE_INCREMENT
    amount = 2 * excess_balance
    state.balances[validator_index] += excess_balance

    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount,
    )

    yield from run_withdrawal_request_processing(
        spec,
        state,
        withdrawal_request,
    )

    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(
        current_epoch
    )


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_pending_withdrawals(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    r