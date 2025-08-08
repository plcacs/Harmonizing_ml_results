from eth2spec import Spec
from eth2spec.test.helpers.forks import is_post_electra
from eth2spec.test.helpers.state import BeaconState
from eth2spec.test.helpers.random import PRNG
from eth2spec.test.helpers.pending_withdrawal import PendingPartialWithdrawal
from eth2spec.test.helpers.withdrawal_request import WithdrawalRequest

def get_expected_withdrawals(spec: Spec, state: BeaconState) -> list:
    ...

def set_validator_fully_withdrawable(spec: Spec, state: BeaconState, index: int, withdrawable_epoch: int = None) -> None:
    ...

def set_eth1_withdrawal_credential_with_balance(spec: Spec, state: BeaconState, index: int, balance: int = None, address: bytes = None) -> None:
    ...

def set_validator_partially_withdrawable(spec: Spec, state: BeaconState, index: int, excess_balance: int = 1000000000) -> None:
    ...

def sample_withdrawal_indices(spec: Spec, state: BeaconState, rng: PRNG, num_full_withdrawals: int, num_partial_withdrawals: int) -> tuple:
    ...

def prepare_expected_withdrawals(spec: Spec, state: BeaconState, rng: PRNG, num_full_withdrawals: int = 0, num_partial_withdrawals: int = 0, num_full_withdrawals_comp: int = 0, num_partial_withdrawals_comp: int = 0) -> tuple:
    ...

def set_compounding_withdrawal_credential(spec: Spec, state: BeaconState, index: int, address: bytes = None) -> None:
    ...

def set_compounding_withdrawal_credential_with_balance(spec: Spec, state: BeaconState, index: int, effective_balance: int = None, balance: int = None, address: bytes = None) -> None:
    ...

def prepare_pending_withdrawal(spec: Spec, state: BeaconState, validator_index: int, effective_balance: int = 32000000000, amount: int = 1000000000, withdrawable_epoch: int = None) -> PendingPartialWithdrawal:
    ...

def prepare_withdrawal_request(spec: Spec, state: BeaconState, validator_index: int, address: bytes = None, amount: int = None) -> WithdrawalRequest:
    ...

def verify_post_state(state: BeaconState, spec: Spec, expected_withdrawals: list, fully_withdrawable_indices: list, partial_withdrawals_indices: list) -> None:
    ...

def run_withdrawals_processing(spec: Spec, state: BeaconState, execution_payload, num_expected_withdrawals: int = None, fully_withdrawable_indices: list = None, partial_withdrawals_indices: list = None, pending_withdrawal_requests: list = None, valid: bool = True) -> list:
    ...
