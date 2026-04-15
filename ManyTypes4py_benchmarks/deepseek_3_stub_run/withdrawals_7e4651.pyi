from typing import Any, Optional, Tuple, List, Generator
from eth2spec.test.helpers.forks import is_post_electra

def get_expected_withdrawals(spec: Any, state: Any) -> Any: ...

def set_validator_fully_withdrawable(
    spec: Any,
    state: Any,
    index: int,
    withdrawable_epoch: Optional[int] = None
) -> None: ...

def set_eth1_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: int,
    balance: Optional[int] = None,
    address: Optional[bytes] = None
) -> None: ...

def set_validator_partially_withdrawable(
    spec: Any,
    state: Any,
    index: int,
    excess_balance: int = 1000000000
) -> None: ...

def sample_withdrawal_indices(
    spec: Any,
    state: Any,
    rng: Any,
    num_full_withdrawals: int,
    num_partial_withdrawals: int
) -> Tuple[List[int], List[int]]: ...

def prepare_expected_withdrawals(
    spec: Any,
    state: Any,
    rng: Any,
    num_full_withdrawals: int = 0,
    num_partial_withdrawals: int = 0,
    num_full_withdrawals_comp: int = 0,
    num_partial_withdrawals_comp: int = 0
) -> Tuple[List[int], List[int]]: ...

def set_compounding_withdrawal_credential(
    spec: Any,
    state: Any,
    index: int,
    address: Optional[bytes] = None
) -> None: ...

def set_compounding_withdrawal_credential_with_balance(
    spec: Any,
    state: Any,
    index: int,
    effective_balance: Optional[int] = None,
    balance: Optional[int] = None,
    address: Optional[bytes] = None
) -> None: ...

def prepare_pending_withdrawal(
    spec: Any,
    state: Any,
    validator_index: int,
    effective_balance: int = 32000000000,
    amount: int = 1000000000,
    withdrawable_epoch: Optional[int] = None
) -> Any: ...

def prepare_withdrawal_request(
    spec: Any,
    state: Any,
    validator_index: int,
    address: Optional[bytes] = None,
    amount: Optional[int] = None
) -> Any: ...

def verify_post_state(
    state: Any,
    spec: Any,
    expected_withdrawals: List[Any],
    fully_withdrawable_indices: List[int],
    partial_withdrawals_indices: List[int]
) -> None: ...

def run_withdrawals_processing(
    spec: Any,
    state: Any,
    execution_payload: Any,
    num_expected_withdrawals: Optional[int] = None,
    fully_withdrawable_indices: Optional[List[int]] = None,
    partial_withdrawals_indices: Optional[List[int]] = None,
    pending_withdrawal_requests: Optional[List[Any]] = None,
    valid: bool = True
) -> Generator[Tuple[str, Optional[Any]], None, Optional[Any]]: ...