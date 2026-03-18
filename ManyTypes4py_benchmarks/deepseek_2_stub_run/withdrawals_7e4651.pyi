```python
from typing import Any, Tuple, List, Generator

def is_post_electra(spec: Any) -> bool: ...

def get_expected_withdrawals(spec: Any, state: Any) -> Any: ...

def set_validator_fully_withdrawable(
    spec: Any,
    state: Any,
    index: Any,
    withdrawable_epoch: Any = ...
) -> None: ...

def set_eth1_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: Any,
    balance: Any = ...,
    address: Any = ...
) -> None: ...

def set_validator_partially_withdrawable(
    spec: Any,
    state: Any,
    index: Any,
    excess_balance: Any = ...
) -> None: ...

def sample_withdrawal_indices(
    spec: Any,
    state: Any,
    rng: Any,
    num_full_withdrawals: Any,
    num_partial_withdrawals: Any
) -> Tuple[Any, Any]: ...

def prepare_expected_withdrawals(
    spec: Any,
    state: Any,
    rng: Any,
    num_full_withdrawals: Any = ...,
    num_partial_withdrawals: Any = ...,
    num_full_withdrawals_comp: Any = ...,
    num_partial_withdrawals_comp: Any = ...
) -> Tuple[Any, Any]: ...

def set_compounding_withdrawal_credential(
    spec: Any,
    state: Any,
    index: Any,
    address: Any = ...
) -> None: ...

def set_compounding_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: Any,
    effective_balance: Any = ...,
    balance: Any = ...,
    address: Any = ...
) -> None: ...

def prepare_pending_withdrawal(
    spec: Any,
    state: Any,
    validator_index: Any,
    effective_balance: Any = ...,
    amount: Any = ...,
    withdrawable_epoch: Any = ...
) -> Any: ...

def prepare_withdrawal_request(
    spec: Any,
    state: Any,
    validator_index: Any,
    address: Any = ...,
    amount: Any = ...
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
    num_expected_withdrawals: Any = ...,
    fully_withdrawable_indices: Any = ...,
    partial_withdrawals_indices: Any = ...,
    pending_withdrawal_requests: Any = ...,
    valid: bool = ...
) -> Generator[Tuple[str, Any], None, Any]: ...
```