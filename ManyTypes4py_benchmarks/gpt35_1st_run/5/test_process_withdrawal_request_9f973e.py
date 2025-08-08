from typing import Iterator
from eth2spec import spec as spec
from eth2spec.test.helpers.constants import MINIMAL
from eth2spec.test.helpers.state import get_validator_index_by_pubkey
from eth2spec.test.helpers.withdrawals import set_eth1_withdrawal_credential_with_balance, set_compounding_withdrawal_credential

def run_withdrawal_request_processing(spec: spec, state: spec.BeaconState, withdrawal_request: spec.WithdrawalRequest, valid: bool = True, success: bool = True) -> Iterator:
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
            expected_partial_withdrawal = spec.PendingPartialWithdrawal(validator_index=validator_index, amount=expected_amount_to_withdraw, withdrawable_epoch=expected_withdrawable_epoch)
            assert state.pending_partial_withdrawals == pre_pending_partial_withdrawals + [expected_partial_withdrawal]

def compute_amount_to_withdraw(spec: spec, state: spec.BeaconState, index: int, amount: int) -> int:
    pending_balance_to_withdraw = spec.get_pending_balance_to_withdraw(state, index)
    return min(state.balances[index] - spec.MIN_ACTIVATION_BALANCE - pending_balance_to_withdraw, amount)
