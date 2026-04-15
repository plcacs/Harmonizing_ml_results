from typing import Any, List, Optional, Tuple, Generator, Union
from random import Random
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.forks import is_post_altair, is_post_electra
from eth2spec.test.helpers.keys import pubkeys, privkeys
from eth2spec.test.helpers.state import get_balance
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_to
from eth2spec.utils import bls
from eth2spec.utils.merkle_minimal import calc_merkle_tree_from_leaves, get_merkle_proof
from eth2spec.utils.ssz.ssz_impl import hash_tree_root
from eth2spec.utils.ssz.ssz_typing import List as SSZList

def mock_deposit(spec: Any, state: Any, index: int) -> None: ...

def build_deposit_data(
    spec: Any,
    pubkey: Any,
    privkey: Any,
    amount: int,
    withdrawal_credentials: bytes,
    fork_version: Optional[bytes] = None,
    signed: bool = False
) -> Any: ...

def sign_deposit_data(
    spec: Any,
    deposit_data: Any,
    privkey: Any,
    fork_version: Optional[bytes] = None
) -> None: ...

def build_deposit(
    spec: Any,
    deposit_data_list: List[Any],
    pubkey: Any,
    privkey: Any,
    amount: int,
    withdrawal_credentials: bytes,
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
    min_pubkey_index: int = 0,
    signed: bool = False,
    deposit_data_list: Optional[List[Any]] = None
) -> Tuple[List[Any], Any, List[Any]]: ...

def prepare_random_genesis_deposits(
    spec: Any,
    deposit_count: int,
    max_pubkey_index: int,
    min_pubkey_index: int = 0,
    max_amount: Optional[int] = None,
    min_amount: Optional[int] = None,
    deposit_data_list: Optional[List[Any]] = None,
    rng: Random = ...
) -> Tuple[List[Any], Any, List[Any]]: ...

def prepare_state_and_deposit(
    spec: Any,
    state: Any,
    validator_index: int,
    amount: int,
    pubkey: Optional[Any] = None,
    privkey: Optional[Any] = None,
    withdrawal_credentials: Optional[bytes] = None,
    signed: bool = False
) -> Any: ...

def prepare_deposit_request(
    spec: Any,
    validator_index: int,
    amount: int,
    index: Optional[int] = None,
    pubkey: Optional[Any] = None,
    privkey: Optional[Any] = None,
    withdrawal_credentials: Optional[bytes] = None,
    signed: bool = False
) -> Any: ...

def prepare_pending_deposit(
    spec: Any,
    validator_index: int,
    amount: int,
    pubkey: Optional[Any] = None,
    privkey: Optional[Any] = None,
    withdrawal_credentials: Optional[bytes] = None,
    fork_version: Optional[bytes] = None,
    signed: bool = False,
    slot: Optional[int] = None
) -> Any: ...

def run_deposit_processing(
    spec: Any,
    state: Any,
    deposit: Any,
    validator_index: int,
    valid: bool = True,
    effective: bool = True
) -> Generator[Tuple[str, Optional[Any]], None, None]: ...

def run_deposit_processing_with_specific_fork_version(
    spec: Any,
    state: Any,
    fork_version: bytes,
    valid: bool = True,
    effective: bool = True
) -> Generator[Tuple[str, Optional[Any]], None, None]: ...

def run_deposit_request_processing(
    spec: Any,
    state: Any,
    deposit_request: Any,
    validator_index: int,
    effective: bool = True
) -> Generator[Tuple[str, Any], None, None]: ...

def run_pending_deposit_applying(
    spec: Any,
    state: Any,
    pending_deposit: Any,
    validator_index: int,
    effective: bool = True
) -> Generator[Tuple[str, Any], None, None]: ...