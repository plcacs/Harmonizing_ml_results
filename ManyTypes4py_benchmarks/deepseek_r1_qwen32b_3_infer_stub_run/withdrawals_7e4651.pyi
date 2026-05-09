from typing import Any, List, Optional, Tuple
from eth2spec.specs.electra.phase0.types import (
    Withdrawal,
    PendingPartialWithdrawal,
    WithdrawalRequest,
    Validator,
    Spec,
    Bytes20,
    Epoch,
    Gwei,
    ValidatorIndex,
)

def get_expected_withdrawals(spec: Spec, state: Any) -> List[Withdrawal]:
    ...

def set_validator_fully_withdrawable(spec: Spec, state: Any, index: int, withdrawable_epoch: Optional[Epoch] = None) -> None:
    ...

def set_eth1_withdrawal_credential_with_balance(spec: Spec, state: Any, index: int, balance: Optional[Gwei] = None, address: Optional[Bytes20] = None) -> None:
    ...

def set_validator_partially_withdrawable(spec: Spec, state: Any, index: int, excess_balance: int = 1000000000) -> None:
    ...

def sample_withdrawal_indices(spec: Spec, state: Any, rng: Any, num_full_withdrawals: int, num_partial_withdrawals: int) -> Tuple[List[int], List[int]]:
    ...

def prepare_expected_withdrawals(spec: Spec, state: Any, rng: Any, num_full_withdrawals: int = 0, num_partial_withdrawals: int = 0, num_full_withdrawals_comp: int = 0, num_partial_withdrawals_comp: int = 0) -> Tuple[List[int], List[int]]:
    ...

def set_compounding_withdrawal_credential(spec: Spec, state: Any, index: int, address: Optional[Bytes20] = None) -> None:
    ...

def set_compounding_withdrawal_credential_with_balance(spec: Spec, state: Any, index: int, effective_balance: Optional[Gwei] = None, balance: Optional[Gwei] = None, address: Optional[Bytes20] = None) -> None:
    ...

def prepare_pending_withdrawal(spec: Spec, state: Any, validator_index: ValidatorIndex, effective_balance: Gwei = 32000000000, amount: Gwei = 1000000000, withdrawable_epoch: Optional[Epoch] = None) -> PendingPartialWithdrawal:
    ...

def prepare_withdrawal_request(spec: Spec, state: Any, validator_index: ValidatorIndex, address: Optional[Bytes20] = None, amount: Optional[Gwei] = None) -> WithdrawalRequest:
    ...

def verify_post_state(state: Any, spec: Spec, expected_withdrawals: List[Withdrawal], fully_withdrawable_indices: List[int], partial_withdrawals_indices: List[int]) -> None:
    ...

def run_withdrawals_processing(spec: Spec, state: Any, execution_payload: Any, num_expected_withdrawals: Optional[int] = None, fully_withdrawable_indices: Optional[List[int]] = None, partial_withdrawals_indices: Optional[List[int]] = None, pending_withdrawal_requests: Optional[List[WithdrawalRequest]] = None, valid: bool = True) -> Optional[List[Withdrawal]]:
    ...