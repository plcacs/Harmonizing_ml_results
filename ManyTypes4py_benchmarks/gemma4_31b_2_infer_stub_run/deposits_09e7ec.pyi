from typing import Any, Generator, Optional, Tuple, Union, List as TypingList
from random import Random

# Assuming spec is a protocol or a base class for the Ethereum spec
# Since it's a complex object, we use Any or a generic TypeVar, 
# but based on usage, it has specific attributes and methods.
Spec = Any 
State = Any

def mock_deposit(spec: Spec, state: State, index: int) -> None: ...

def build_deposit_data(
    spec: Spec,
    pubkey: bytes,
    privkey: bytes,
    amount: int,
    withdrawal_credentials: bytes,
    fork_version: Optional[int] = None,
    signed: bool = False,
) -> Any: ...

def sign_deposit_data(
    spec: Spec,
    deposit_data: Any,
    privkey: bytes,
    fork_version: Optional[int] = None,
) -> None: ...

def build_deposit(
    spec: Spec,
    deposit_data_list: TypingList[Any],
    pubkey: bytes,
    privkey: bytes,
    amount: int,
    withdrawal_credentials: bytes,
    signed: bool,
) -> Tuple[Any, bytes, TypingList[Any]]: ...

def deposit_from_context(spec: Spec, deposit_data_list: TypingList[Any], index: int) -> Tuple[Any, bytes, TypingList[Any]]: ...

def prepare_full_genesis_deposits(
    spec: Spec,
    amount: int,
    deposit_count: int,
    min_pubkey_index: int = 0,
    signed: bool = False,
    deposit_data_list: Optional[TypingList[Any]] = None,
) -> Tuple[TypingList[Any], bytes, TypingList[Any]]: ...

def prepare_random_genesis_deposits(
    spec: Spec,
    deposit_count: int,
    max_pubkey_index: int,
    min_pubkey_index: int = 0,
    max_amount: Optional[int] = None,
    min_amount: Optional[int] = None,
    deposit_data_list: Optional[TypingList[Any]] = None,
    rng: Random = ...,
) -> Tuple[TypingList[Any], bytes, TypingList[Any]]: ...

def prepare_state_and_deposit(
    spec: Spec,
    state: State,
    validator_index: int,
    amount: int,
    pubkey: Optional[bytes] = None,
    privkey: Optional[bytes] = None,
    withdrawal_credentials: Optional[bytes] = None,
    signed: bool = False,
) -> Any: ...

def prepare_deposit_request(
    spec: Spec,
    validator_index: int,
    amount: int,
    index: Optional[int] = None,
    pubkey: Optional[bytes] = None,
    privkey: Optional[bytes] = None,
    withdrawal_credentials: Optional[bytes] = None,
    signed: bool = False,
) -> Any: ...

def prepare_pending_deposit(
    spec: Spec,
    validator_index: int,
    amount: int,
    pubkey: Optional[bytes] = None,
    privkey: Optional[bytes] = None,
    withdrawal_credentials: Optional[bytes] = None,
    fork_version: Optional[int] = None,
    signed: bool = False,
    slot: Optional[int] = None,
) -> Any: ...

def run_deposit_processing(
    spec: Spec,
    state: State,
    deposit: Any,
    validator_index: int,
    valid: bool = True,
    effective: bool = True,
) -> Generator[Tuple[str, Optional[State]], None, None]: ...

def run_deposit_processing_with_specific_fork_version(
    spec: Spec,
    state: State,
    fork_version: int,
    valid: bool = True,
    effective: bool = True,
) -> Generator[Tuple[str, Optional[State]], None, None]: ...

def run_deposit_request_processing(
    spec: Spec,
    state: State,
    deposit_request: Any,
    validator_index: int,
    effective: bool = True,
) -> Generator[Tuple[str, Union[State, Any]], None, None]: ...

def run_pending_deposit_applying(
    spec: Spec,
    state: State,
    pending_deposit: Any,
    validator_index: int,
    effective: bool = True,
) -> Generator[Tuple[str, State], None, None]: ...