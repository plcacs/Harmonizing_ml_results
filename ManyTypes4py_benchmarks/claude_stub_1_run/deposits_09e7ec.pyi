```python
from random import Random
from typing import Any, Generator, Optional, Tuple, List as ListType

def mock_deposit(spec: Any, state: Any, index: int) -> None: ...

def build_deposit_data(
    spec: Any,
    pubkey: Any,
    privkey: Any,
    amount: int,
    withdrawal_credentials: Any,
    fork_version: Optional[Any] = None,
    signed: bool = False,
) -> Any: ...

def sign_deposit_data(
    spec: Any,
    deposit_data: Any,
    privkey: Any,
    fork_version: Optional[Any] = None,
) -> None: ...

def build_deposit(
    spec: Any,
    deposit_data_list: ListType[Any],
    pubkey: Any,
    privkey: Any,
    amount: int,
    withdrawal_credentials: Any,
    signed: bool,
) -> Tuple[Any, Any, ListType[Any]]: ...

def deposit_from_context(
    spec: Any,
    deposit_data_list: ListType[Any],
    index: int,
) -> Tuple[Any, Any, ListType[Any]]: ...

def prepare_full_genesis_deposits(
    spec: Any,
    amount: int,
    deposit_count: int,
    min_pubkey_index: int = 0,
    signed: bool = False,
    deposit_data_list: Optional[ListType[Any]] = None,
) -> Tuple[ListType[Any], Any, ListType[Any]]: ...

def prepare_random_genesis_deposits(
    spec: Any,
    deposit_count: int,
    max_pubkey_index: int,
    min_pubkey_index: int = 0,
    max_amount: Optional[int] = None,
    min_amount: Optional[int] = None,
    deposit_data_list: Optional[ListType[Any]] = None,
    rng: Random = ...,
) -> Tuple[ListType[Any], Any, ListType[Any]]: ...

def prepare_state_and_deposit(
    spec: Any,
    state: Any,
    validator_index: int,
    amount: int,
    pubkey: Optional[Any] = None,
    privkey: Optional[Any] = None,
    withdrawal_credentials: Optional[Any] = None,
    signed: bool = False,
) -> Any: ...

def prepare_deposit_request(
    spec: Any,
    validator_index: int,
    amount: int,
    index: Optional[int] = None,
    pubkey: Optional[Any] = None,
    privkey: Optional[Any] = None,
    withdrawal_credentials: Optional[Any] = None,
    signed: bool = False,
) -> Any: ...

def prepare_pending_deposit(
    spec: Any,
    validator_index: int,
    amount: int,
    pubkey: Optional[Any] = None,
    privkey: Optional[Any] = None,
    withdrawal_credentials: Optional[Any] = None,
    fork_version: Optional[Any] = None,
    signed: bool = False,
    slot: Optional[int] = None,
) -> Any: ...

def run_deposit_processing(
    spec: Any,
    state: Any,
    deposit: Any,
    validator_index: int,
    valid: bool = True,
    effective: bool = True,
) -> Generator[Tuple[str, Any], None, None]: ...

def run_deposit_processing_with_specific_fork_version(
    spec: Any,
    state: Any,
    fork_version: Any,
    valid: bool = True,
    effective: bool = True,
) -> Generator[Tuple[str, Any], None, None]: ...

def run_deposit_request_processing(
    spec: Any,
    state: Any,
    deposit_request: Any,
    validator_index: int,
    effective: bool = True,
) -> Generator[Tuple[str, Any], None, None]: ...

def run_pending_deposit_applying(
    spec: Any,
    state: Any,
    pending_deposit: Any,
    validator_index: int,
    effective: bool = True,
) -> Generator[Tuple[str, Any], None, None]: ...
```