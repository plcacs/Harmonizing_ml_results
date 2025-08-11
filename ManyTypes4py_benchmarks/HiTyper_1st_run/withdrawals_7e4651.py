from eth2spec.test.helpers.forks import is_post_electra

def get_expected_withdrawals(spec: dict["core.Edge", "state.State"], state: Union[dict["core.Edge", "state.State"], dict]) -> Union[list, bool, dict, str, State]:
    if is_post_electra(spec):
        withdrawals, _ = spec.get_expected_withdrawals(state)
        return withdrawals
    else:
        return spec.get_expected_withdrawals(state)

def set_validator_fully_withdrawable(spec: Union[int, dict, float], state: Union[dict, int, None], index: Union[int, str, typing.Type], withdrawable_epoch: Union[None, int, float]=None) -> None:
    if withdrawable_epoch is None:
        withdrawable_epoch = spec.get_current_epoch(state)
    validator = state.validators[index]
    validator.withdrawable_epoch = withdrawable_epoch
    if validator.exit_epoch > withdrawable_epoch:
        validator.exit_epoch = withdrawable_epoch
    if validator.withdrawal_credentials[0:1] == spec.BLS_WITHDRAWAL_PREFIX:
        validator.withdrawal_credentials = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + validator.withdrawal_credentials[1:]
    if state.balances[index] == 0:
        state.balances[index] = 10000000000
    assert spec.is_fully_withdrawable_validator(validator, state.balances[index], withdrawable_epoch)

def set_eth1_withdrawal_credential_with_balance(spec: Any, state: Any, index: Any, balance: Union[None, bool]=None, address: None=None) -> None:
    if balance is None:
        balance = spec.MAX_EFFECTIVE_BALANCE
    if address is None:
        address = b'\x11' * 20
    validator = state.validators[index]
    validator.withdrawal_credentials = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + address
    validator.effective_balance = min(balance, spec.MAX_EFFECTIVE_BALANCE)
    state.balances[index] = balance

def set_validator_partially_withdrawable(spec: Any, state: Union[int, typing.Callable, str], index: Union[int, typing.Callable, str], excess_balance: int=1000000000) -> None:
    validator = state.validators[index]
    if is_post_electra(spec) and spec.has_compounding_withdrawal_credential(validator):
        validator.effective_balance = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
        state.balances[index] = validator.effective_balance + excess_balance
    else:
        set_eth1_withdrawal_credential_with_balance(spec, state, index, spec.MAX_EFFECTIVE_BALANCE + excess_balance)
    assert spec.is_partially_withdrawable_validator(state.validators[index], state.balances[index])

def sample_withdrawal_indices(spec: Union[int, tuple[int]], state: Union[int, tuple[int]], rng: Union[int, tuple[int], list[int]], num_full_withdrawals: int, num_partial_withdrawals: int) -> tuple[list]:
    bound = min(len(state.validators), spec.MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP)
    assert num_full_withdrawals + num_partial_withdrawals <= bound
    eligible_validator_indices = list(range(bound))
    sampled_indices = rng.sample(eligible_validator_indices, num_full_withdrawals + num_partial_withdrawals)
    fully_withdrawable_indices = rng.sample(sampled_indices, num_full_withdrawals)
    partial_withdrawals_indices = list(set(sampled_indices).difference(set(fully_withdrawable_indices)))
    return (fully_withdrawable_indices, partial_withdrawals_indices)

def prepare_expected_withdrawals(spec: Union[int, list[int], dict, None], state: Union[int, list[int], dict, None], rng: Union[int, list[int], dict, None], num_full_withdrawals: int=0, num_partial_withdrawals: int=0, num_full_withdrawals_comp: int=0, num_partial_withdrawals_comp: int=0) -> tuple[set[int]]:
    fully_withdrawable_indices, partial_withdrawals_indices = sample_withdrawal_indices(spec, state, rng, num_full_withdrawals + num_full_withdrawals_comp, num_partial_withdrawals + num_partial_withdrawals_comp)
    fully_withdrawable_indices_comp = rng.sample(fully_withdrawable_indices, num_full_withdrawals_comp)
    partial_withdrawals_indices_comp = rng.sample(partial_withdrawals_indices, num_partial_withdrawals_comp)
    for index in fully_withdrawable_indices_comp + partial_withdrawals_indices_comp:
        address = state.validators[index].withdrawal_credentials[12:]
        set_compounding_withdrawal_credential_with_balance(spec, state, index, address=address)
    for index in fully_withdrawable_indices:
        set_validator_fully_withdrawable(spec, state, index)
    for index in partial_withdrawals_indices:
        set_validator_partially_withdrawable(spec, state, index)
    return (fully_withdrawable_indices, partial_withdrawals_indices)

def set_compounding_withdrawal_credential(spec: Union[str, typing.ClassVar, bytes], state: Any, index: Any, address: Union[None, dict, str]=None) -> None:
    if address is None:
        address = b'\x11' * 20
    validator = state.validators[index]
    validator.withdrawal_credentials = spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + address

def set_compounding_withdrawal_credential_with_balance(spec: int, state: Any, index: Any, effective_balance: Union[None, dict[str, int], tuple[int]]=None, balance: Union[None, dict[str, int], tuple[int], str]=None, address: None=None) -> None:
    set_compounding_withdrawal_credential(spec, state, index, address)
    if effective_balance is None:
        effective_balance = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    if balance is None:
        balance = effective_balance
    state.validators[index].effective_balance = effective_balance
    state.balances[index] = balance

def prepare_pending_withdrawal(spec: Union[int, dict], state: Union[int, list[int], collections.abc.Awaitable[None]], validator_index: int, effective_balance: int=32000000000, amount: int=1000000000, withdrawable_epoch: Union[None, int, typing.Mapping]=None) -> Union[list, int, bytearray]:
    assert is_post_electra(spec)
    if withdrawable_epoch is None:
        withdrawable_epoch = spec.get_current_epoch(state)
    balance = effective_balance + amount
    set_compounding_withdrawal_credential_with_balance(spec, state, validator_index, effective_balance, balance)
    withdrawal = spec.PendingPartialWithdrawal(validator_index=validator_index, amount=amount, withdrawable_epoch=withdrawable_epoch)
    state.pending_partial_withdrawals.append(withdrawal)
    return withdrawal

def prepare_withdrawal_request(spec: Any, state: Any, validator_index: Any, address: Union[None, str, int]=None, amount: Union[None, int]=None):
    validator = state.validators[validator_index]
    if not spec.has_execution_withdrawal_credential(validator):
        set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)
    if amount is None:
        amount = spec.FULL_EXIT_REQUEST_AMOUNT
    return spec.WithdrawalRequest(source_address=state.validators[validator_index].withdrawal_credentials[12:], validator_pubkey=state.validators[validator_index].pubkey, amount=amount)

def verify_post_state(state: Union[list[int], int, list[set[int]]], spec: Union[list[float], list[int], list[str]], expected_withdrawals: Union[list[int], list[list[int]]], fully_withdrawable_indices: Union[list[int], tuple[str]], partial_withdrawals_indices: Union[list[int], tuple[str]]) -> None:
    if len(expected_withdrawals) == 0:
        return
    expected_withdrawals_validator_indices = [withdrawal.validator_index for withdrawal in expected_withdrawals]
    assert state.next_withdrawal_index == expected_withdrawals[-1].index + 1
    if len(expected_withdrawals) == spec.MAX_WITHDRAWALS_PER_PAYLOAD:
        next_withdrawal_validator_index = (expected_withdrawals_validator_indices[-1] + 1) % len(state.validators)
        assert state.next_withdrawal_validator_index == next_withdrawal_validator_index
    for index in fully_withdrawable_indices:
        if index in expected_withdrawals_validator_indices:
            assert state.balances[index] == 0
        else:
            assert state.balances[index] > 0
    for index in partial_withdrawals_indices:
        if is_post_electra(spec):
            max_effective_balance = spec.get_max_effective_balance(state.validators[index])
        else:
            max_effective_balance = spec.MAX_EFFECTIVE_BALANCE
        if index in expected_withdrawals_validator_indices:
            assert state.balances[index] == max_effective_balance
        else:
            assert state.balances[index] > max_effective_balance

def run_withdrawals_processing(spec: Union[int, dict], state: dict[str, typing.Any], execution_payload: Union[dict, int, bytes], num_expected_withdrawals: Union[None, int]=None, fully_withdrawable_indices: Union[None, str, int, tuple[int]]=None, partial_withdrawals_indices: Union[None, str, int, tuple[int]]=None, pending_withdrawal_requests: None=None, valid: bool=True) -> Union[typing.Generator[tuple[typing.Union[typing.Text,dict[str, typing.Any]]]], typing.Generator[tuple[typing.Union[typing.Text,dict,int,bytes]]], typing.Generator[tuple[typing.Optional[typing.Text]]], None, dict, dict[str, int]]:
    """
    Run ``process_withdrawals``, yielding:
      - pre-state ('pre')
      - execution payload ('execution_payload')
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    expected_withdrawals = get_expected_withdrawals(spec, state)
    assert len(expected_withdrawals) <= spec.MAX_WITHDRAWALS_PER_PAYLOAD
    if num_expected_withdrawals is not None:
        assert len(expected_withdrawals) == num_expected_withdrawals
    pre_state = state.copy()
    yield ('pre', state)
    yield ('execution_payload', execution_payload)
    if not valid:
        try:
            spec.process_withdrawals(state, execution_payload)
            raise AssertionError('expected an assertion error, but got none.')
        except AssertionError:
            pass
        yield ('post', None)
        return
    spec.process_withdrawals(state, execution_payload)
    yield ('post', state)
    assert state.next_withdrawal_index == pre_state.next_withdrawal_index + len(expected_withdrawals)
    for index, withdrawal in enumerate(execution_payload.withdrawals):
        assert withdrawal.index == pre_state.next_withdrawal_index + index
    if len(expected_withdrawals) == 0:
        next_withdrawal_validator_index = pre_state.next_withdrawal_validator_index + spec.MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP
        assert state.next_withdrawal_validator_index == next_withdrawal_validator_index % len(state.validators)
    elif len(expected_withdrawals) <= spec.MAX_WITHDRAWALS_PER_PAYLOAD:
        bound = min(spec.MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP, spec.MAX_WITHDRAWALS_PER_PAYLOAD)
        assert len(get_expected_withdrawals(spec, state)) <= bound
    elif len(expected_withdrawals) > spec.MAX_WITHDRAWALS_PER_PAYLOAD:
        raise ValueError('len(expected_withdrawals) should not be greater than MAX_WITHDRAWALS_PER_PAYLOAD')
    if fully_withdrawable_indices is not None or partial_withdrawals_indices is not None:
        verify_post_state(state, spec, expected_withdrawals, fully_withdrawable_indices, partial_withdrawals_indices)
    if pending_withdrawal_requests is not None:
        assert len(pending_withdrawal_requests) <= len(execution_payload.withdrawals)
        for index, request in enumerate(pending_withdrawal_requests):
            withdrawal = execution_payload.withdrawals[index]
            assert withdrawal.validator_index == request.validator_index
            assert withdrawal.amount == request.amount
    return expected_withdrawals