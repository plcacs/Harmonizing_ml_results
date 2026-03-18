```python
from typing import Any, Tuple, List, Iterator, Optional
from eth2spec.test.helpers.forks import is_post_electra

def get_expected_withdrawals(spec: Any, state: Any) -> Any: ...

def set_validator_fully_withdrawable(
    spec: Any,
    state: Any,
    index: Any,
    withdrawable_epoch: Any = None
) -> None: ...

def set_eth1_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: Any,
    balance: Any = None,
    address: Any = None
) -> None: ...

def set_validator_partially_withdrawable(
    spec: Any,
    state: Any,
    index: Any,
    excess_balance: Any = 1000000000
) -> None: ...

def sample_withdrawal_indices(
    spec: Any,
    state: Any,
    rng: Any,
    num_full_withdrawals: Any,
    num_partial_withdrawals: Any
) -> Tuple[List[Any], List[Any]]: ...

def prepare_expected_withdrawals(
    spec: Any,
    state: Any,
    rng: Any,
    num_full_withdrawals: Any = 0,
    num_partial_withdrawals: Any = 0,
    num_full_withdrawals_comp: Any = 0,
    num_partial_withdrawals_comp: Any = 0
) -> Tuple[List[Any], List[Any]]: ...

def set_compounding_withdrawal_credential(
    spec: Any,
    state: Any,
    index: Any,
    address: Any = None
) -> None: ...

def set_compounding_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: Any,
    effective_balance: Any = None,
    balance: Any = None,
    address: Any = None
) -> None: ...

def prepare_pending_withdrawal(
    spec: Any,
    state: Any,
    validator_index: Any,
    effective_balance: Any = 32000000000,
    amount: Any = 1000000000,
    withdrawable_epoch: Any = None
) -> Any: ...

def prepare_withdrawal_request(
    spec: Any,
    state: Any,
    validator_index: Any,
    address: Any = None,
    amount: Any = None
) -> Any: ...

def verify_post_state(
    state: Any,
    spec: Any,
    expected_withdrawals: Any,
    fully_withdrawable_indices: Any,
    partial_withdrawals_indices: Any
) -> None: ...

def run_withdrawals_processing(
    spec: Any,
    state: Any,
    execution_payload: Any,
    num_expected_withdrawals: Any = None,
    fully_withdrawable_indices: Any = None,
    partial_withdrawals_indices: Any = None,
    pending_withdrawal_requests: Any = None,
    valid: bool = True
) -> Iterator[Tuple[str, Optional[Any]]]: ...
```