```python
from typing import Any, Iterator, List, Set, Tuple, Optional
from random import Random

def run_slash_and_exit(
    spec: Any,
    state: Any,
    slash_index: Any,
    exit_index: Any,
    valid: bool = True
) -> Iterator[Tuple[str, Any]]: ...

def get_random_proposer_slashings(
    spec: Any,
    state: Any,
    rng: Any
) -> List[Any]: ...

def get_random_attester_slashings(
    spec: Any,
    state: Any,
    rng: Any,
    slashed_indices: List[Any] = ...
) -> List[Any]: ...

def get_random_attestations(
    spec: Any,
    state: Any,
    rng: Any
) -> List[Any]: ...

def get_random_deposits(
    spec: Any,
    state: Any,
    rng: Any,
    num_deposits: Optional[int] = None
) -> Tuple[List[Any], bytes]: ...

def prepare_state_and_get_random_deposits(
    spec: Any,
    state: Any,
    rng: Any,
    num_deposits: Optional[int] = None
) -> List[Any]: ...

def _eligible_for_exit(
    spec: Any,
    state: Any,
    index: Any
) -> bool: ...

def get_random_voluntary_exits(
    spec: Any,
    state: Any,
    to_be_slashed_indices: Set[Any],
    rng: Any
) -> List[Any]: ...

def get_random_sync_aggregate(
    spec: Any,
    state: Any,
    slot: Any,
    block_root: Any = None,
    fraction_participated: float = 1.0,
    rng: Random = ...
) -> Any: ...

def get_random_bls_to_execution_changes(
    spec: Any,
    state: Any,
    rng: Random = ...,
    num_address_changes: int = 0
) -> List[Any]: ...

def build_random_block_from_state_for_next_slot(
    spec: Any,
    state: Any,
    rng: Random = ...,
    deposits: Optional[List[Any]] = None
) -> Any: ...

def run_test_full_random_operations(
    spec: Any,
    state: Any,
    rng: Random = ...
) -> Iterator[Tuple[str, Any]]: ...

def get_random_execution_requests(
    spec: Any,
    state: Any,
    rng: Any
) -> Any: ...

def get_random_deposit_requests(
    spec: Any,
    state: Any,
    rng: Any,
    num_deposits: Optional[int] = None
) -> List[Any]: ...

def get_random_withdrawal_requests(
    spec: Any,
    state: Any,
    rng: Any,
    num_withdrawals: Optional[int] = None
) -> List[Any]: ...

def get_random_consolidation_requests(
    spec: Any,
    state: Any,
    rng: Any,
    num_consolidations: Optional[int] = None
) -> List[Any]: ...
```