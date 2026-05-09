from typing import Any, Generator, Iterable, List, Optional, Tuple, Set, Union
from random import Random

# Assuming these types based on the usage in the provided module
# Since the actual classes are from eth2spec, we use Any or generic names 
# where the specific class definition isn't provided, but we maintain 
# the structure of the spec and state objects.

# Type aliases for clarity based on the domain (Ethereum 2.0 / eth2spec)
Spec = Any
State = Any
Block = Any
SignedBlock = Any
Validator = Any
Deposit = Any
ProposerSlashing = Any
AttesterSlashing = Any
Attestation = Any
SignedExit = Any
SyncAggregate = Any
ExecutionRequests = Any
DepositRequest = Any
WithdrawalRequest = Any
ConsolidationRequest = Any

def run_slash_and_exit(
    spec: Spec, 
    state: State, 
    slash_index: int, 
    exit_index: int, 
    valid: bool = True
) -> Generator[Tuple[str, Optional[Union[State, List[SignedBlock]]]], None, None]: ...

def get_random_proposer_slashings(spec: Spec, state: State, rng: Random) -> List[ProposerSlashing]: ...

def get_random_attester_slashings(
    spec: Spec, 
    state: State, 
    rng: Random, 
    slashed_indices: List[int] = []
) -> List[AttesterSlashing]: ...

def get_random_attestations(spec: Spec, state: State, rng: Random) -> List[Attestation]: ...

def get_random_deposits(
    spec: Spec, 
    state: State, 
    rng: Random, 
    num_deposits: Optional[int] = None
) -> Tuple[List[Deposit], bytes]: ...

def prepare_state_and_get_random_deposits(
    spec: Spec, 
    state: State, 
    rng: Random, 
    num_deposits: Optional[int] = None
) -> List[Deposit]: ...

def _eligible_for_exit(spec: Spec, state: State, index: int) -> bool: ...

def get_random_voluntary_exits(
    spec: Spec, 
    state: State, 
    to_be_slashed_indices: Union[List[int], Set[int]], 
    rng: Random
) -> List[SignedExit]: ...

def get_random_sync_aggregate(
    spec: Spec, 
    state: State, 
    slot: int, 
    block_root: Optional[bytes] = None, 
    fraction_participated: float = 1.0, 
    rng: Random = ...
) -> SyncAggregate: ...

def get_random_bls_to_execution_changes(
    spec: Spec, 
    state: State, 
    rng: Random = ..., 
    num_address_changes: int = 0
) -> List[Any]: ...

def build_random_block_from_state_for_next_slot(
    spec: Spec, 
    state: State, 
    rng: Random = ..., 
    deposits: Optional[List[Deposit]] = None
) -> Block: ...

def run_test_full_random_operations(
    spec: Spec, 
    state: State, 
    rng: Random = ...
) -> Generator[Tuple[str, Optional[Union[State, List[SignedBlock]]]], None, None]: ...

def get_random_execution_requests(spec: Spec, state: State, rng: Random) -> ExecutionRequests: ...

def get_random_deposit_requests(
    spec: Spec, 
    state: State, 
    rng: Random, 
    num_deposits: Optional[int] = None
) -> List[DepositRequest]: ...

def get_random_withdrawal_requests(
    spec: Spec, 
    state: State, 
    rng: Random, 
    num_withdrawals: Optional[int] = None
) -> List[WithdrawalRequest]: ...

def get_random_consolidation_requests(
    spec: Spec, 
    state: State, 
    rng: Random, 
    num_consolidations: Optional[int] = None
) -> List[ConsolidationRequest]: ...