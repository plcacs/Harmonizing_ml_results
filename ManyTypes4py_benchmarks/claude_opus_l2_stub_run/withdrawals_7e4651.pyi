from typing import Any, Generator, Optional
import random

def get_expected_withdrawals(spec: Any, state: Any) -> Any: ...

def set_validator_fully_withdrawable(
    spec: Any,
    state: Any,
    index: int,
    withdrawable_epoch: Optional[Any] = ...,
) -> None: ...

def set_eth1_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: int,
    balance: Optional[int] = ...,
    address: Optional[bytes] = ...,
) -> None: ...

def set_validator_partially_withdrawable(
    spec: Any,
    state: Any,
    index: int,
    excess_balance: int = ...,
) -> None: ...

def sample_withdrawal_indices(
    spec: Any,
    state: Any,
    rng: random.Random,
    num_full_withdrawals: int,
    num_partial_withdrawals: int,
) -> tuple[list[int], list[int]]: ...

def prepare_expected_withdrawals(
    spec: Any,
    state: Any,
    rng: random.Random,
    num_full_withdrawals: int = ...,
    num_partial_withdrawals: int = ...,
    num_full_withdrawals_comp: int = ...,
    num_partial_withdrawals_comp: int = ...,
) -> tuple[list[int], list[int]]: ...

def set_compounding_withdrawal_credential(
    spec: Any,
    state: Any,
    index: int,
    address: Optional[bytes] = ...,
) -> None: ...

def set_compounding_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: int,
    effective_balance: Optional[int] = ...,
    balance: Optional[int] = ...,
    address: Optional[bytes] = ...,
) -> None: ...

def prepare_pending_withdrawal(
    spec: Any,
    state: Any,
    validator_index: int,
    effective_balance: int = ...,
    amount: int = ...,
    withdrawable_epoch: Optional[Any] = ...,
) -> Any: ...

def prepare_withdrawal_request(
    spec: Any,
    state: Any,
    validator_index: int,
    address: Optional[bytes] = ...,
    amount: Optional[int] = ...,
) -> Any: ...

def verify_post_state(
    state: Any,
    spec: Any,
    expected_withdrawals: Any,
    fully_withdrawable_indices: list[int],
    partial_withdrawals_indices: list[int],
) -> None: ...

def run_withdrawals_processing(
    spec: Any,
    state: Any,
    execution_payload: Any,
    num_expected_withdrawals: Optional[int] = ...,
    fully_withdrawable_indices: Optional[list[int]] = ...,
    partial_withdrawals_indices: Optional[list[int]] = ...,
    pending_withdrawal_requests: Optional[list[Any]] = ...,
    valid: bool = ...,
) -> Generator[tuple[str, Any], None, Optional[Any]]: ...