```python
from typing import Any, List, Tuple, Iterator
from random import Random

def mock_deposit(spec: Any, state: Any, index: int) -> None: ...

def build_deposit_data(
    spec: Any,
    pubkey: Any,
    privkey: Any,
    amount: int,
    withdrawal_credentials: Any,
    fork_version: Any = ...,
    signed: bool = False
) -> Any: ...

def sign_deposit_data(
    spec: Any,
    deposit_data: Any,
    privkey: Any,
    fork_version: Any = ...
) -> None: ...

def build_deposit(
    spec: Any,
    deposit_data_list: List[Any],
    pubkey: Any,
    privkey: Any,
    amount: int,
    withdrawal_credentials: Any,
    signed: bool
) -> Tuple[Any, Any, List[Any]]: ...

def deposit_from_context(
    spec: Any,
    deposit_data_list: List[Any],
    index: int
) -> Tuple[Any, Any, List[Any]]: ...

def prepare_full_genesis_deposits(
    spec: Any,
    amount: int,
    deposit_count: int,
    min_pubkey_index: int = ...,
    signed: bool = False,
    deposit_data_list: List[Any] = ...
) -> Tuple[List[Any], Any, List[Any]]: ...

def prepare_random_genesis_deposits(
    spec: Any,
    deposit_count: int,
    max_pubkey_index: int,
    min_pubkey_index: int = ...,
    max_amount: int = ...,
    min_amount: int = ...,
    deposit_data_list: List[Any] = ...,
    rng: Random = ...
) -> Tuple[List[Any], Any, List[Any]]: ...

def prepare_state_and_deposit(
    spec: Any,
    state: Any,
    validator_index: int,
    amount: int,
    pubkey: Any = ...,
    privkey: Any = ...,
    withdrawal_credentials: Any = ...,
    signed: bool = False
) -> Any: ...

def prepare_deposit_request(
    spec: Any,
    validator_index: int,
    amount: int,
    index: int = ...,
    pubkey: Any = ...,
    privkey: Any = ...,
    withdrawal_credentials: Any = ...,
    signed: bool = False
) -> Any: ...

def prepare_pending_deposit(
    spec: Any,
    validator_index: int,
    amount: int,
    pubkey: Any = ...,
    privkey: Any = ...,
    withdrawal_credentials: Any = ...,
    fork_version: Any = ...,
    signed: bool = False,
    slot: int = ...
) -> Any: ...

def run_deposit_processing(
    spec: Any,
    state: Any,
    deposit: Any,
    validator_index: int,
    valid: bool = ...,
    effective: bool = ...
) -> Iterator[Tuple[str, Any]]: ...

def run_deposit_processing_with_specific_fork_version(
    spec: Any,
    state: Any,
    fork_version: Any,
    valid: bool = ...,
    effective: bool = ...
) -> Iterator[Tuple[str, Any]]: ...

def run_deposit_request_processing(
    spec: Any,
    state: Any,
    deposit_request: Any,
    validator_index: int,
    effective: bool = ...
) -> Iterator[Tuple[str, Any]]: ...

def run_pending_deposit_applying(
    spec: Any,
    state: Any,
    pending_deposit: Any,
    validator_index: int,
    effective: bool = ...
) -> Iterator[Tuple[str, Any]]: ...
```