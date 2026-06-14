from random import Random
from typing import Any, Generator

def run_slash_and_exit(
    spec: Any,
    state: Any,
    slash_index: int,
    exit_index: int,
    valid: bool = True,
) -> Generator[tuple[str, Any], None, None]: ...

def get_random_proposer_slashings(spec: Any, state: Any, rng: Random) -> list[Any]: ...

def get_random_attester_slashings(
    spec: Any,
    state: Any,
    rng: Random,
    slashed_indices: list[Any] = ...,
) -> list[Any]: ...

def get_random_attestations(spec: Any, state: Any, rng: Random) -> list[Any]: ...

def get_random_deposits(
    spec: Any,
    state: Any,
    rng: Random,
    num_deposits: int | None = None,
) -> tuple[list[Any], Any]: ...

def prepare_state_and_get_random_deposits(
    spec: Any,
    state: Any,
    rng: Random,
    num_deposits: int | None = None,
) -> list[Any]: ...

def _eligible_for_exit(spec: Any, state: Any, index: int) -> bool: ...

def get_random_voluntary_exits(
    spec: Any,
    state: Any,
    to_be_slashed_indices: set[Any],
    rng: Random,
) -> list[Any]: ...

def get_random_sync_aggregate(
    spec: Any,
    state: Any,
    slot: int,
    block_root: Any = None,
    fraction_participated: float = 1.0,
    rng: Random = ...,
) -> Any: ...

def get_random_bls_to_execution_changes(
    spec: Any,
    state: Any,
    rng: Random = ...,
    num_address_changes: int = 0,
) -> list[Any]: ...

def build_random_block_from_state_for_next_slot(
    spec: Any,
    state: Any,
    rng: Random = ...,
    deposits: list[Any] | None = None,
) -> Any: ...

def run_test_full_random_operations(
    spec: Any,
    state: Any,
    rng: Random = ...,
) -> Generator[tuple[str, Any], None, None]: ...

def get_random_execution_requests(spec: Any, state: Any, rng: Random) -> Any: ...

def get_random_deposit_requests(
    spec: Any,
    state: Any,
    rng: Random,
    num_deposits: int | None = None,
) -> list[Any]: ...

def get_random_withdrawal_requests(
    spec: Any,
    state: Any,
    rng: Random,
    num_withdrawals: int | None = None,
) -> list[Any]: ...

def get_random_consolidation_requests(
    spec: Any,
    state: Any,
    rng: Random,
    num_consolidations: int | None = None,
) -> list[Any]: ...