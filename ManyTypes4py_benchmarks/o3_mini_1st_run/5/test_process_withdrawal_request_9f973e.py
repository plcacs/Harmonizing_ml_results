import random
from typing import Any, Iterator, Tuple, Optional
from eth2spec.test.context import spec_state_test, expect_assertion_error, with_electra_and_later, with_presets
from eth2spec.test.helpers.constants import MINIMAL
from eth2spec.test.helpers.state import get_validator_index_by_pubkey
from eth2spec.test.helpers.withdrawals import set_eth1_withdrawal_credential_with_balance, set_compounding_withdrawal_credential

def run_withdrawal_request_processing(
    spec: Any,
    state: Any,
    withdrawal_request: Any,
    valid: bool = True,
    success: bool = True
) -> Iterator[Tuple[str, Optional[Any]]]:
    """
    Run ``process_withdrawal_request``, yielding:
      - pre-state ('pre')
      - withdrawal_request ('withdrawal_request')
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    If ``success == False``, it doesn't initiate exit successfully
    """
    yield ('pre', state)
    yield ('withdrawal_request', withdrawal_request)
    if not valid:
        expect_assertion_error(lambda: spec.process_withdrawal_request(state, withdrawal_request))
        yield ('post', None)
        return
    pre_state = state.copy()
    spec.process_withdrawal_request(state, withdrawal_request)
    yield ('post', state)
    if not success:
        assert pre_state == state
    else:
        validator_index = get_validator_index_by_pubkey(state, withdrawal_request.validator_pubkey)
        pre_exit_epoch = pre_state.validators[validator_index].exit_epoch
        pre_pending_partial_withdrawals = pre_state.pending_partial_withdrawals.copy()
        pre_balance = pre_state.balances[validator_index]
        pre_effective_balance = pre_state.validators[validator_index].effective_balance
        assert state.balances[validator_index] == pre_balance
        assert state.validators[validator_index].effective_balance == pre_effective_balance
        if withdrawal_request.amount == spec.FULL_EXIT_REQUEST_AMOUNT:
            assert pre_exit_epoch == spec.FAR_FUTURE_EPOCH
            assert state.validators[validator_index].exit_epoch < spec.FAR_FUTURE_EPOCH
            assert spec.get_pending_balance_to_withdraw(state, validator_index) == 0
            assert state.pending_partial_withdrawals == pre_pending_partial_withdrawals
        else:
            expected_amount_to_withdraw = compute_amount_to_withdraw(spec, pre_state, validator_index, withdrawal_request.amount)
            assert state.validators[validator_index].exit_epoch == spec.FAR_FUTURE_EPOCH
            expected_withdrawable_epoch = state.earliest_exit_epoch + spec.config.MIN_VALIDATOR_WITHDRAWABILITY_DELAY
            expected_partial_withdrawal = spec.PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=expected_amount_to_withdraw,
                withdrawable_epoch=expected_withdrawable_epoch
            )
            assert state.pending_partial_withdrawals == pre_pending_partial_withdrawals + [expected_partial_withdrawal]

def compute_amount_to_withdraw(spec: Any, state: Any, index: int, amount: int) -> int:
    pending_balance_to_withdraw = spec.get_pending_balance_to_withdraw(state, index)
    return min(state.balances[index] - spec.MIN_ACTIVATION_BALANCE - pending_balance_to_withdraw, amount)

@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1337)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)

@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request_with_first_validator(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = spec.get_active_validator_indices(state, current_epoch)[0]
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)

@with_electra_and_later
@spec_state_test
def test_basic_withdrawal_request_with_compounding_credentials(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1338)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], 'need full partial withdrawal queue')
def test_basic_withdrawal_request_with_full_partial_withdrawal_queue(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1339)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    partial_withdrawal: Any = spec.PendingPartialWithdrawal(
        validator_index=1,
        amount=1,
        withdrawable_epoch=current_epoch
    )
    state.pending_partial_withdrawals = [partial_withdrawal] * spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)

@with_electra_and_later
@spec_state_test
def test_incorrect_source_address(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1340)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    incorrect_address: bytes = b'3' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=incorrect_address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_incorrect_withdrawal_credential_prefix(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1341)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    state.validators[validator_index].withdrawal_credentials = (
        spec.BLS_WITHDRAWAL_PREFIX + state.validators[validator_index].withdrawal_credentials[1:]
    )
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_on_withdrawal_request_initiated_exit_validator(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1342)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    spec.initiate_validator_exit(state, validator_index)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_activation_epoch_less_than_shard_committee_period(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1343)
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    assert spec.get_current_epoch(state) < state.validators[validator_index].activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_unknown_pubkey(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1344)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    address: bytes = b'"' * 20
    pubkey: Any = spec.BLSPubkey(b'#' * 48)
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1344)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)
    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(current_epoch)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request_higher_excess_balance(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1345)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)
    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(current_epoch)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_basic_partial_withdrawal_request_lower_than_excess_balance(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1346)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    excess_balance: int = spec.EFFECTIVE_BALANCE_INCREMENT
    amount: int = 2 * excess_balance
    state.balances[validator_index] += excess_balance
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)
    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(current_epoch)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_pending_withdrawals(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1347)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    partial_withdrawal: Any = spec.PendingPartialWithdrawal(
        validator_index=validator_index,
        amount=amount,
        withdrawable_epoch=current_epoch
    )
    state.pending_partial_withdrawals = [partial_withdrawal] * 2
    state.balances[validator_index] += 3 * amount
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)
    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(current_epoch)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_pending_withdrawals_and_high_amount(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1348)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.UINT64_MAX
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    partial_withdrawal: Any = spec.PendingPartialWithdrawal(
        validator_index=validator_index,
        amount=spec.EFFECTIVE_BALANCE_INCREMENT,
        withdrawable_epoch=current_epoch
    )
    state.pending_partial_withdrawals = [partial_withdrawal] * (spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT - 1)
    state.balances[validator_index] = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_high_balance(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1349)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    state.balances[validator_index] = 3 * spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    state.validators[validator_index].effective_balance = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    churn_limit: int = spec.get_activation_exit_churn_limit(state)
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)
    exit_epoch: int = spec.compute_activation_exit_epoch(current_epoch) + amount // churn_limit
    assert state.earliest_exit_epoch == exit_epoch

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_high_amount(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1350)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.UINT64_MAX
    state.balances[validator_index] += 1
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)
    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(current_epoch)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL])
def test_partial_withdrawal_request_with_low_amount(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1351)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = 1
    state.balances[validator_index] += amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request)
    assert state.earliest_exit_epoch == spec.compute_activation_exit_epoch(current_epoch)

@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], 'need full partial withdrawal queue')
def test_partial_withdrawal_queue_full(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1352)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    partial_withdrawal: Any = spec.PendingPartialWithdrawal(
        validator_index=1,
        amount=1,
        withdrawable_epoch=current_epoch
    )
    state.pending_partial_withdrawals = [partial_withdrawal] * spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_no_compounding_credentials(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1353)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_no_excess_balance(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1354)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_pending_withdrawals_consume_all_excess_balance(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1355)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 10 * amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    partial_withdrawal: Any = spec.PendingPartialWithdrawal(
        validator_index=validator_index,
        amount=amount,
        withdrawable_epoch=current_epoch
    )
    state.pending_partial_withdrawals = [partial_withdrawal] * 10
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_insufficient_effective_balance(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1356)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.validators[validator_index].effective_balance -= spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += spec.EFFECTIVE_BALANCE_INCREMENT
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_incorrect_source_address(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1357)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    incorrect_address: bytes = b'3' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=incorrect_address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_incorrect_withdrawal_credential_prefix(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1358)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    state.validators[validator_index].withdrawal_credentials = (
        spec.BLS_WITHDRAWAL_PREFIX + state.validators[validator_index].withdrawal_credentials[1:]
    )
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_on_exit_initiated_validator(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1359)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    spec.initiate_validator_exit(state, validator_index)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_partial_withdrawal_activation_epoch_less_than_shard_committee_period(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1360)
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    state.balances[validator_index] += 2 * amount
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    assert spec.get_current_epoch(state) < state.validators[validator_index].activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_insufficient_balance(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1361)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    amount: int = spec.EFFECTIVE_BALANCE_INCREMENT
    set_compounding_withdrawal_credential(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=amount
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_full_exit_request_has_partial_withdrawal(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1361)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    state.balances[validator_index] = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    state.pending_partial_withdrawals.append(
        spec.PendingPartialWithdrawal(
            validator_index=validator_index,
            amount=1,
            withdrawable_epoch=spec.compute_activation_exit_epoch(current_epoch)
        )
    )
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)

@with_electra_and_later
@spec_state_test
def test_incorrect_inactive_validator(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    rng: random.Random = random.Random(1361)
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    current_epoch: int = spec.get_current_epoch(state)
    validator_index: int = rng.choice(spec.get_active_validator_indices(state, current_epoch))
    validator_pubkey: Any = state.validators[validator_index].pubkey
    address: bytes = b'"' * 20
    set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    withdrawal_request: Any = spec.WithdrawalRequest(
        source_address=address,
        validator_pubkey=validator_pubkey,
        amount=spec.FULL_EXIT_REQUEST_AMOUNT
    )
    state.validators[validator_index].activation_epoch = spec.FAR_FUTURE_EPOCH
    assert not spec.is_active_validator(state.validators[validator_index], current_epoch)
    yield from run_withdrawal_request_processing(spec, state, withdrawal_request, success=False)