```python
from typing import Any, List as TypingList, Tuple, Iterator, Optional
from random import Random
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.forks import is_post_altair, is_post_electra
from eth2spec.test.helpers.keys import pubkeys, privkeys
from eth2spec.test.helpers.state import get_balance
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_to
from eth2spec.utils import bls
from eth2spec.utils.merkle_minimal import calc_merkle_tree_from_leaves, get_merkle_proof
from eth2spec.utils.ssz.ssz_impl import hash_tree_root
from eth2spec.utils.ssz.ssz_typing import List

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
    deposit_data_list: TypingList[Any],
    pubkey: Any,
    privkey: Any,
    amount: int,
    withdrawal_credentials: Any,
    signed: bool
) -> Tuple[Any, Any, TypingList[Any]]: ...

def deposit_from_context(
    spec: Any,
    deposit_data_list: TypingList[Any],
    index: int
) -> Tuple[Any, Any, TypingList[Any]]: ...

def prepare_full_genesis_deposits(
    spec: Any,
    amount: int,
    deposit_count: int,
    min_pubkey_index: int = ...,
    signed: bool = False,
    deposit_data_list: Optional[TypingList[Any]] = ...
) -> Tuple[TypingList[Any], Any, TypingList[Any]]: ...

def prepare_random_genesis_deposits(
    spec: Any,
    deposit_count: int,
    max_pubkey_index: int,
    min_pubkey_index: int = ...,
    max_amount: Optional[int] = ...,
    min_amount: Optional[int] = ...,
    deposit_data_list: Optional[TypingList[Any]] = ...,
    rng: Random = ...
) -> Tuple[TypingList[Any], Any, TypingList[Any]]: ...

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
    index: Optional[int] = ...,
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
    slot: Optional[int] = ...
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