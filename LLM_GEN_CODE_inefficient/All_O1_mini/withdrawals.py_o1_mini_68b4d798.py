from eth2spec.test.helpers.forks import is_post_electra
from typing import List, Tuple, Optional, Generator, Union
from random import Random


# Assuming Spec, State, Withdrawal, ExecutionPayload, WithdrawalRequest, PendingPartialWithdrawal are defined elsewhere
class Spec:
    BLS_WITHDRAWAL_PREFIX: bytes
    ETH1_ADDRESS_WITHDRAWAL_PREFIX: bytes
    COMPOUNDING_WITHDRAWAL_PREFIX: bytes
    MAX_EFFECTIVE_BALANCE: int
    MAX_EFFECTIVE_BALANCE_ELECTRA: int
    MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP: int
    MAX_WITHDRAWALS_PER_PAYLOAD: int
    FULL_EXIT_REQUEST_AMOUNT: int

    def get_expected_withdrawals(self, state: 'State') -> Tuple[List['Withdrawal'], any]:
        ...

    def get_current_epoch(self, state: 'State') -> int:
        ...

    def is_fully_withdrawable_validator(self, validator: 'Validator', balance: int, withdrawable_epoch: int) -> bool:
        ...

    def has_compounding_withdrawal_credential(self, validator: 'Validator') -> bool:
        ...

    def is_partially_withdrawable_validator(self, validator: 'Validator', balance: int) -> bool:
        ...

    def process_withdrawals(self, state: 'State', execution_payload: 'ExecutionPayload') -> None:
        ...

    def get_max_effective_balance(self, validator: 'Validator') -> int:
        ...

    def has_execution_withdrawal_credential(self, validator: 'Validator') -> bool:
        ...


class Validator:
    withdrawal_credentials: bytes
    exit_epoch: int
    withdrawable_epoch: int
    effective_balance: int
    pubkey: bytes


class State:
    validators: List[Validator]
    balances: List[int]
    next_withdrawal_index: int
    next_withdrawal_validator_index: int
    pending_partial_withdrawals: List['PendingPartialWithdrawal']

    def copy(self) -> 'State':
        ...


class Withdrawal:
    validator_index: int
    index: int


class ExecutionPayload:
    withdrawals: List[Withdrawal]


class WithdrawalRequest:
    source_address: bytes
    validator_pubkey: bytes
    amount: int


class PendingPartialWithdrawal:
    validator_index: int
    amount: int
    withdrawable_epoch: int


def get_expected_withdrawals(spec: Spec, state: State) -> List[Withdrawal]:
    if is_post_electra(spec):
        withdrawals, _ = spec.get_expected_withdrawals(state)
        return withdrawals
    else:
        return spec.get_expected_withdrawals(state)


def set_validator_fully_withdrawable(spec: Spec, state: State, index: int, withdrawable_epoch: Optional[int] = None) -> None:
    if withdrawable_epoch is None:
        withdrawable_epoch = spec.get_current_epoch(state)

    validator: Validator = state.validators[index]
    validator.withdrawable_epoch = withdrawable_epoch
    # set exit epoch as well to avoid interactions with other epoch process, e.g. forced ejections
    if validator.exit_epoch > withdrawable_epoch:
        validator.exit_epoch = withdrawable_epoch

    if validator.withdrawal_credentials[0:1] == spec.BLS_WITHDRAWAL_PREFIX:
        validator.withdrawal_credentials = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + validator.withdrawal_credentials[1:]

    if state.balances[index] == 0:
        state.balances[index] = 10000000000

    assert spec.is_fully_withdrawable_validator(validator, state.balances[index], withdrawable_epoch)


def set_eth1_withdrawal_credential_with_balance(
    spec: Spec, state: State, index: int, balance: Optional[int] = None, address: Optional[bytes] = None
) -> None:
    if balance is None:
        balance = spec.MAX_EFFECTIVE_BALANCE
    if address is None:
        address = b'\x11' * 20

    validator: Validator = state.validators[index]
    validator.withdrawal_credentials = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + address
    validator.effective_balance = min(balance, spec.MAX_EFFECTIVE_BALANCE)
    state.balances[index] = balance


def set_validator_partially_withdrawable(
    spec: Spec, state: State, index: int, excess_balance: int = 1000000000
) -> None:
    validator: Validator = state.validators[index]
    if is_post_electra(spec) and spec.has_compounding_withdrawal_credential(validator):
        validator.effective_balance = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
        state.balances[index] = validator.effective_balance + excess_balance
    else:
        set_eth1_withdrawal_credential_with_balance(
            spec, state, index, spec.MAX_EFFECTIVE_BALANCE + excess_balance
        )

    assert spec.is_partially_withdrawable_validator(state.validators[index], state.balances[index])


def sample_withdrawal_indices(
    spec: Spec, state: State, rng: Random, num_full_withdrawals: int, num_partial_withdrawals: int
) -> Tuple[List[int], List[int]]:
    bound: int = min(len(state.validators), spec.MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP)
    assert num_full_withdrawals + num_partial_withdrawals <= bound
    eligible_validator_indices: List[int] = list(range(bound))
    sampled_indices: List[int] = rng.sample(eligible_validator_indices, num_full_withdrawals + num_partial_withdrawals)
    fully_withdrawable_indices: List[int] = rng.sample(sampled_indices, num_full_withdrawals)
    partial_withdrawals_indices: List[int] = list(set(sampled_indices).difference(set(fully_withdrawable_indices)))

    return fully_withdrawable_indices, partial_withdrawals_indices


def prepare_expected_withdrawals(
    spec: Spec,
    state: State,
    rng: Random,
    num_full_withdrawals: int = 0,
    num_partial_withdrawals: int = 0,
    num_full_withdrawals_comp: int = 0,
    num_partial_withdrawals_comp: int = 0
) -> Tuple[List[int], List[int]]:
    fully_withdrawable_indices, partial_withdrawals_indices = sample_withdrawal_indices(
        spec, state, rng,
        num_full_withdrawals + num_full_withdrawals_comp,
        num_partial_withdrawals + num_partial_withdrawals_comp
    )

    fully_withdrawable_indices_comp: List[int] = rng.sample(fully_withdrawable_indices, num_full_withdrawals_comp)
    partial_withdrawals_indices_comp: List[int] = rng.sample(partial_withdrawals_indices, num_partial_withdrawals_comp)

    for index in (fully_withdrawable_indices_comp + partial_withdrawals_indices_comp):
        address: bytes = state.validators[index].withdrawal_credentials[12:]
        set_compounding_withdrawal_credential_with_balance(spec, state, index, address=address)

    for index in fully_withdrawable_indices:
        set_validator_fully_withdrawable(spec, state, index)
    for index in partial_withdrawals_indices:
        set_validator_partially_withdrawable(spec, state, index)

    return fully_withdrawable_indices, partial_withdrawals_indices


def set_compounding_withdrawal_credential(spec: Spec, state: State, index: int, address: Optional[bytes] = None) -> None:
    if address is None:
        address = b'\x11' * 20

    validator: Validator = state.validators[index]
    validator.withdrawal_credentials = spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + address


def set_compounding_withdrawal_credential_with_balance(
    spec: Spec,
    state: State,
    index: int,
    effective_balance: Optional[int] = None,
    balance: Optional[int] = None,
    address: Optional[bytes] = None
) -> None:
    set_compounding_withdrawal_credential(spec, state, index, address)

    if effective_balance is None:
        effective_balance = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    if balance is None:
        balance = effective_balance

    state.validators[index].effective_balance = effective_balance
    state.balances[index] = balance


def prepare_pending_withdrawal(
    spec: Spec,
    state: State,
    validator_index: int,
    effective_balance: int = 32_000_000_000,
    amount: int = 1_000_000_000,
    withdrawable_epoch: Optional[int] = None
) -> PendingPartialWithdrawal:
    assert is_post_electra(spec)

    if withdrawable_epoch is None:
        withdrawable_epoch = spec.get_current_epoch(state)

    balance: int = effective_balance + amount
    set_compounding_withdrawal_credential_with_balance(
        spec, state, validator_index, effective_balance, balance
    )

    withdrawal = PendingPartialWithdrawal(
        validator_index=validator_index,
        amount=amount,
        withdrawable_epoch=withdrawable_epoch,
    )
    state.pending_partial_withdrawals.append(withdrawal)

    return withdrawal


def prepare_withdrawal_request(
    spec: Spec,
    state: State,
    validator_index: int,
    address: Optional[bytes] = None,
    amount: Optional[int] = None
) -> WithdrawalRequest:
    validator: Validator = state.validators[validator_index]
    if not spec.has_execution_withdrawal_credential(validator):
        set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)

    if amount is None:
        amount = spec.FULL_EXIT_REQUEST_AMOUNT

    return WithdrawalRequest(
        source_address=state.validators[validator_index].withdrawal_credentials[12:],
        validator_pubkey=state.validators[validator_index].pubkey,
        amount=amount,
    )


#
# Run processing
#


def verify_post_state(
    state: State,
    spec: Spec,
    expected_withdrawals: List[Withdrawal],
    fully_withdrawable_indices: List[int],
    partial_withdrawals_indices: List[int]
) -> None:
    # Consider verifying also the condition when no withdrawals are expected.
    if len(expected_withdrawals) == 0:
        return

    expected_withdrawals_validator_indices: List[int] = [
        withdrawal.validator_index for withdrawal in expected_withdrawals
    ]
    assert state.next_withdrawal_index == expected_withdrawals[-1].index + 1

    if len(expected_withdrawals) == spec.MAX_WITHDRAWALS_PER_PAYLOAD:
        # NOTE: ideally we would also check in the case with
        # fewer than maximum withdrawals but that requires the pre-state info
        next_withdrawal_validator_index: int = (
            expected_withdrawals_validator_indices[-1] + 1
        ) % len(state.validators)
        assert state.next_withdrawal_validator_index == next_withdrawal_validator_index

    for index in fully_withdrawable_indices:
        if index in expected_withdrawals_validator_indices:
            assert state.balances[index] == 0
        else:
            assert state.balances[index] > 0
    for index in partial_withdrawals_indices:
        if is_post_electra(spec):
            max_effective_balance: int = spec.get_max_effective_balance(state.validators[index])
        else:
            max_effective_balance = spec.MAX_EFFECTIVE_BALANCE

        if index in expected_withdrawals_validator_indices:
            assert state.balances[index] == max_effective_balance
        else:
            assert state.balances[index] > max_effective_balance


def run_withdrawals_processing(
    spec: Spec,
    state: State,
    execution_payload: ExecutionPayload,
    num_expected_withdrawals: Optional[int] = None,
    fully_withdrawable_indices: Optional[List[int]] = None,
    partial_withdrawals_indices: Optional[List[int]] = None,
    pending_withdrawal_requests: Optional[List[WithdrawalRequest]] = None,
    valid: bool = True
) -> Generator[Tuple[str, Union[State, ExecutionPayload, None]], None, Optional[List[Withdrawal]]]:
    """
    Run ``process_withdrawals``, yielding:
      - pre-state ('pre')
      - execution payload ('execution_payload')
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    expected_withdrawals: List[Withdrawal] = get_expected_withdrawals(spec, state)
    assert len(expected_withdrawals) <= spec.MAX_WITHDRAWALS_PER_PAYLOAD
    if num_expected_withdrawals is not None:
        assert len(expected_withdrawals) == num_expected_withdrawals

    pre_state: State = state.copy()
    yield 'pre', state
    yield 'execution_payload', execution_payload

    if not valid:
        try:
            spec.process_withdrawals(state, execution_payload)
            raise AssertionError('expected an assertion error, but got none.')
        except AssertionError:
            pass

        yield 'post', None
        return

    spec.process_withdrawals(state, execution_payload)

    yield 'post', state

    # Check withdrawal indices
    assert state.next_withdrawal_index == pre_state.next_withdrawal_index + len(expected_withdrawals)
    for index, withdrawal in enumerate(execution_payload.withdrawals):
        assert withdrawal.index == pre_state.next_withdrawal_index + index

    if len(expected_withdrawals) == 0:
        next_withdrawal_validator_index: int = (
            pre_state.next_withdrawal_validator_index + spec.MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP
        )
        assert state.next_withdrawal_validator_index == next_withdrawal_validator_index % len(state.validators)
    elif len(expected_withdrawals) <= spec.MAX_WITHDRAWALS_PER_PAYLOAD:
        bound: int = min(spec.MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP, spec.MAX_WITHDRAWALS_PER_PAYLOAD)
        assert len(get_expected_withdrawals(spec, state)) <= bound
    elif len(expected_withdrawals) > spec.MAX_WITHDRAWALS_PER_PAYLOAD:
        raise ValueError('len(expected_withdrawals) should not be greater than MAX_WITHDRAWALS_PER_PAYLOAD')

    if fully_withdrawable_indices is not None or partial_withdrawals_indices is not None:
        verify_post_state(state, spec, expected_withdrawals, fully_withdrawable_indices or [], partial_withdrawals_indices or [])

    # Check withdrawal requests
    if pending_withdrawal_requests is not None:
        assert len(pending_withdrawal_requests) <= len(execution_payload.withdrawals)
        for index, request in enumerate(pending_withdrawal_requests):
            withdrawal: Withdrawal = execution_payload.withdrawals[index]
            assert withdrawal.validator_index == request.validator_index
            assert withdrawal.amount == request.amount

    return expected_withdrawals
