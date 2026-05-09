from typing import Any, Generator, Optional, Tuple, Union, Sequence, TypeVar, overload
from typing_extensions import Protocol

# Define Protocols for the complex objects used in the module to ensure type safety
# based on the attribute access patterns observed in the code.

class SpecProtocol(Protocol):
    BLS_WITHDRAWAL_PREFIX: bytes
    ETH1_ADDRESS_WITHDRAWAL_PREFIX: bytes
    COMPOUNDING_WITHDRAWAL_PREFIX: bytes
    MAX_EFFECTIVE_BALANCE: int
    MAX_EFFECTIVE_BALANCE_ELECTRA: int
    MAX_VALIDATORS_PER_WITHDRAWALS_SWEEP: int
    MAX_WITHDRAWALS_PER_PAYLOAD: int
    FULL_EXIT_REQUEST_AMOUNT: int
    def get_expected_withdrawals(self, state: Any) -> Union[Sequence[Any], Tuple[Sequence[Any], Any]]: ...
    def get_current_epoch(self, state: Any) -> int: ...
    def is_fully_withdrawable_validator(self, validator: Any, balance: int, epoch: int) -> bool: ...
    def is_partially_withdrawable_validator(self, validator: Any, balance: int) -> bool: ...
    def has_compounding_withdrawal_credential(self, validator: Any) -> bool: ...
    def has_execution_withdrawal_credential(self, validator: Any) -> bool: ...
    def get_max_effective_balance(self, validator: Any) -> int: ...
    def process_withdrawals(self, state: Any, execution_payload: Any) -> None: ...
    def PendingPartialWithdrawal(self, validator_index: int, amount: int, withdrawable_epoch: int) -> Any: ...
    def WithdrawalRequest(self, source_address: bytes, validator_pubkey: bytes, amount: int) -> Any: ...

class ValidatorProtocol(Protocol):
    withdrawable_epoch: int
    exit_epoch: int
    withdrawal_credentials: bytes
    effective_balance: int
    pubkey: bytes

class StateProtocol(Protocol):
    validators: Sequence[ValidatorProtocol]
    balances: list[int]
    pending_partial_withdrawals: list[Any]
    next_withdrawal_index: int
    next_withdrawal_validator_index: int
    def copy(self) -> 'StateProtocol': ...

class ExecutionPayloadProtocol(Protocol):
    withdrawals: Sequence[Any]

class WithdrawalProtocol(Protocol):
    index: int
    validator_index: int
    amount: int

def get_expected_withdrawals(spec: SpecProtocol, state: StateProtocol) -> Sequence[Any]: ...

def set_validator_fully_withdrawable(
    spec: SpecProtocol, 
    state: StateProtocol, 
    index: int, 
    withdrawable_epoch: Optional[int] = None
) -> None: ...

def set_eth1_withdrawal_credential_with_balance(
    spec: SpecProtocol, 
    state: StateProtocol, 
    index: int, 
    balance: Optional[int] = None, 
    address: Optional[bytes] = None
) -> None: ...

def set_validator_partially_withdrawable(
    spec: SpecProtocol, 
    state: StateProtocol, 
    index: int, 
    excess_balance: int = 1000000000
) -> None: ...

def sample_withdrawal_indices(
    spec: SpecProtocol, 
    state: StateProtocol, 
    rng: Any, 
    num_full_withdrawals: int, 
    num_partial_withdrawals: int
) -> Tuple[list[int], list[int]]: ...

def prepare_expected_withdrawals(
    spec: SpecProtocol, 
    state: StateProtocol, 
    rng: Any, 
    num_full_withdrawals: int = 0, 
    num_partial_withdrawals: int = 0, 
    num_full_withdrawals_comp: int = 0, 
    num_partial_withdrawals_comp: int = 0
) -> Tuple[list[int], list[int]]: ...

def set_compounding_withdrawal_credential(
    spec: SpecProtocol, 
    state: StateProtocol, 
    index: int, 
    address: Optional[bytes] = None
) -> None: ...

def set_compounding_withdrawal_credential_with_balance(
    spec: SpecProtocol, 
    state: StateProtocol, 
    index: int, 
    effective_balance: Optional[int] = None, 
    balance: Optional[int] = None, 
    address: Optional[bytes] = None
) -> None: ...

def prepare_pending_withdrawal(
    spec: SpecProtocol, 
    state: StateProtocol, 
    validator_index: int, 
    effective_balance: int = 32000000000, 
    amount: int = 1000000000, 
    withdrawable_epoch: Optional[int] = None
) -> Any: ...

def prepare_withdrawal_request(
    spec: SpecProtocol, 
    state: StateProtocol, 
    validator_index: int, 
    address: Optional[bytes] = None, 
    amount: Optional[int] = None
) -> Any: ...

def verify_post_state(
    state: StateProtocol, 
    spec: SpecProtocol, 
    expected_withdrawals: Sequence[WithdrawalProtocol], 
    fully_withdrawable_indices: Sequence[int], 
    partial_withdrawals_indices: Sequence[int]
) -> None: ...

def run_withdrawals_processing(
    spec: SpecProtocol, 
    state: StateProtocol, 
    execution_payload: ExecutionPayloadProtocol, 
    num_expected_withdrawals: Optional[int] = None, 
    fully_withdrawable_indices: Optional[Sequence[int]] = None, 
    partial_withdrawals_indices: Optional[Sequence[int]] = None, 
    pending_withdrawal_requests: Optional[Sequence[Any]] = None, 
    valid: bool = True
) -> Generator[Tuple[str, Optional[Any]], None, Optional[Sequence[Any]]]: ...