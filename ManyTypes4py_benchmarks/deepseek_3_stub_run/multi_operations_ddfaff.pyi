from typing import Iterator, List, Set, Optional, Tuple, Any
from random import Random

def run_slash_and_exit(
    spec: Any,
    state: Any,
    slash_index: int,
    exit_index: int,
    valid: bool = True
) -> Iterator[Tuple[str, Optional[Any]]]: ...

def get_random_proposer_slashings(
    spec: Any,
    state: Any,
    rng: Random
) -> List[Any]: ...

def get_random_attester_slashings(
    spec: Any,
    state: Any,
    rng: Random,
    slashed_indices: List[int] = []
) -> List[Any]: ...

def get_random_attestations(
    spec: Any,
    state: Any,
    rng: Random
) -> List[Any]: ...

def get_random_deposits(
    spec: Any,
    state: Any,
    rng: Random,
    num_deposits: Optional[int] = None
) -> Tuple[List[Any], bytes]: ...

def prepare_state_and_get_random_deposits(
    spec: Any,
    state: Any,
    rng: Random,
    num_deposits: Optional[int] = None
) -> List[Any]: ...

def _eligible_for_exit(
    spec: Any,
    state: Any,
    index: int
) -> bool: ...

def get_random_voluntary_exits(
    spec: Any,
    state: Any,
    to_be_slashed_indices: Set[int],
    rng: Random
) -> List[Any]: ...

def get_random_sync_aggregate(
    spec: Any,
    state: Any,
    slot: int,
    block_root: Optional[Any] = None,
    fraction_participated: float = 1.0,
    rng: Random = Random(2099)
) -> Any: ...

def get_random_bls_to_execution_changes(
    spec: Any,
    state: Any,
    rng: Random = Random(2188),
    num_address_changes: int = 0
) -> List[Any]: ...

def build_random_block_from_state_for_next_slot(
    spec: Any,
    state: Any,
    rng: Random = Random(2188),
    deposits: Optional[List[Any]] = None
) -> Any: ...

def run_test_full_random_operations(
    spec: Any,
    state: Any,
    rng: Random = Random(2080)
) -> Iterator[Tuple[str, Any]]: ...

def get_random_execution_requests(
    spec: Any,
    state: Any,
    rng: Random
) -> Any: ...

def get_random_deposit_requests(
    spec: Any,
    state: Any,
    rng: Random,
    num_deposits: Optional[int] = None
) -> List[Any]: ...

def get_random_withdrawal_requests(
    spec: Any,
    state: Any,
    rng: Random,
    num_withdrawals: Optional[int] = None
) -> List[Any]: ...

def get_random_consolidation_requests(
    spec: Any,
    state: Any,
    rng: Random,
    num_consolidations: Optional[int] = None
) -> List[Any]: ...