from typing import Any, Callable, Generator, List, Optional, Tuple, Union
from eth2spec.types import (
    DepositData,
    Deposit,
    DepositRequest,
    PendingDeposit,
    ForkVersion,
    BLSPubkey,
    BLSSignature,
    Bytes32,
    Gwei,
    Slot,
    Epoch,
    ValidatorIndex,
)
from random import Random

def mock_deposit(spec: Any, state: Any, index: int) -> None:
    ...

def build_deposit_data(
    spec: Any,
    pubkey: BLSPubkey,
    privkey: BLSSignature,
    amount: Gwei,
    withdrawal_credentials: Bytes32,
    fork_version: Optional[ForkVersion] = None,
    signed: bool = False,
) -> DepositData:
    ...

def sign_deposit_data(
    spec: Any,
    deposit_data: DepositData,
    privkey: BLSSignature,
    fork_version: Optional[ForkVersion] = None,
) -> None:
    ...

def build_deposit(
    spec: Any,
    deposit_data_list: List[DepositData],
    pubkey: BLSPubkey,
    privkey: BLSSignature,
    amount: Gwei,
    withdrawal_credentials: Bytes32,
    signed: bool,
) -> Tuple[Deposit, Bytes32, List[DepositData]]:
    ...

def deposit_from_context(
    spec: Any,
    deposit_data_list: List[DepositData],
    index: int,
) -> Tuple[Deposit, Bytes32, List[DepositData]]:
    ...

def prepare_full_genesis_deposits(
    spec: Any,
    amount: Gwei,
    deposit_count: int,
    min_pubkey_index: int = 0,
    signed: bool = False,
    deposit_data_list: Optional[List[DepositData]] = None,
) -> Tuple[List[Deposit], Bytes32, List[DepositData]]:
    ...

def prepare_random_genesis_deposits(
    spec: Any,
    deposit_count: int,
    max_pubkey_index: int,
    min_pubkey_index: int = 0,
    max_amount: Optional[Gwei] = None,
    min_amount: Optional[Gwei] = None,
    deposit_data_list: Optional[List[DepositData]] = None,
    rng: Random = Random(3131),
) -> Tuple[List[Deposit], Bytes32, List[DepositData]]:
    ...

def prepare_state_and_deposit(
    spec: Any,
    state: Any,
    validator_index: int,
    amount: Gwei,
    pubkey: Optional[BLSPubkey] = None,
    privkey: Optional[BLSSignature] = None,
    withdrawal_credentials: Optional[Bytes32] = None,
    signed: bool = False,
) -> Deposit:
    ...

def prepare_deposit_request(
    spec: Any,
    validator_index: int,
    amount: Gwei,
    index: Optional[int] = None,
    pubkey: Optional[BLSPubkey] = None,
    privkey: Optional[BLSSignature] = None,
    withdrawal_credentials: Optional[Bytes32] = None,
    signed: bool = False,
) -> DepositRequest:
    ...

def prepare_pending_deposit(
    spec: Any,
    validator_index: int,
    amount: Gwei,
    pubkey: Optional[BLSPubkey] = None,
    privkey: Optional[BLSSignature] = None,
    withdrawal_credentials: Optional[Bytes32] = None,
    fork_version: Optional[ForkVersion] = None,
    signed: bool = False,
    slot: Optional[Slot] = None,
) -> PendingDeposit:
    ...

def run_deposit_processing(
    spec: Any,
    state: Any,
    deposit: Deposit,
    validator_index: int,
    valid: bool = True,
    effective: bool = True,
) -> Generator[Tuple[str, Union[Any, None]], None, None]:
    ...

def run_deposit_processing_with_specific_fork_version(
    spec: Any,
    state: Any,
    fork_version: ForkVersion,
    valid: bool = True,
    effective: bool = True,
) -> Generator[Tuple[str, Union[Any, None]], None, None]:
    ...

def run_deposit_request_processing(
    spec: Any,
    state: Any,
    deposit_request: DepositRequest,
    validator_index: int,
    effective: bool = True,
) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_pending_deposit_applying(
    spec: Any,
    state: Any,
    pending_deposit: PendingDeposit,
    validator_index: int,
    effective: bool = True,
) -> Generator[Tuple[str, Any], None, None]:
    ...