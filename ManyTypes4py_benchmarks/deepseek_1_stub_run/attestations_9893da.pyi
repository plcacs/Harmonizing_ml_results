```python
from typing import Any, Callable, Generator, Iterator, List, Optional, Set, Tuple
from lru import LRU
from eth2spec.utils.ssz.ssz_typing import Bitlist

def run_attestation_processing(
    spec: Any,
    state: Any,
    attestation: Any,
    valid: bool = ...
) -> Generator[Tuple[str, Any], None, None]: ...

def build_attestation_data(
    spec: Any,
    state: Any,
    slot: int,
    index: int,
    beacon_block_root: Optional[Any] = ...,
    shard: Optional[Any] = ...
) -> Any: ...

def get_valid_attestation(
    spec: Any,
    state: Any,
    slot: Optional[int] = ...,
    index: Optional[int] = ...,
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = ...,
    beacon_block_root: Optional[Any] = ...,
    signed: bool = ...
) -> Any: ...

def sign_aggregate_attestation(
    spec: Any,
    state: Any,
    attestation_data: Any,
    participants: List[int]
) -> Any: ...

def sign_indexed_attestation(
    spec: Any,
    state: Any,
    indexed_attestation: Any
) -> None: ...

def sign_attestation(
    spec: Any,
    state: Any,
    attestation: Any
) -> None: ...

def get_attestation_signature(
    spec: Any,
    state: Any,
    attestation_data: Any,
    privkey: Any
) -> Any: ...

def compute_max_inclusion_slot(
    spec: Any,
    attestation: Any
) -> int: ...

def fill_aggregate_attestation(
    spec: Any,
    state: Any,
    attestation: Any,
    committee_index: int,
    signed: bool = ...,
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = ...
) -> None: ...

def add_attestations_to_state(
    spec: Any,
    state: Any,
    attestations: List[Any],
    slot: int
) -> None: ...

def get_valid_attestations_at_slot(
    state: Any,
    spec: Any,
    slot_to_attest: int,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = ...,
    beacon_block_root: Optional[Any] = ...
) -> Iterator[Any]: ...

def get_valid_attestation_at_slot(
    state: Any,
    spec: Any,
    slot_to_attest: int,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = ...,
    beacon_block_root: Optional[Any] = ...
) -> Any: ...

def next_slots_with_attestations(
    spec: Any,
    state: Any,
    slot_count: int,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = ...
) -> Tuple[Any, List[Any], Any]: ...

def _add_valid_attestations(
    spec: Any,
    state: Any,
    block: Any,
    slot_to_attest: int,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = ...
) -> None: ...

def next_epoch_with_attestations(
    spec: Any,
    state: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = ...
) -> Tuple[Any, List[Any], Any]: ...

def state_transition_with_full_block(
    spec: Any,
    state: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = ...,
    sync_aggregate: Optional[Any] = ...,
    block: Optional[Any] = ...
) -> Any: ...

def state_transition_with_full_attestations_block(
    spec: Any,
    state: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool
) -> Any: ...

def prepare_state_with_attestations(
    spec: Any,
    state: Any,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = ...
) -> List[Any]: ...

_prep_state_cache_dict: LRU = ...

def cached_prepare_state_with_attestations(
    spec: Any,
    state: Any
) -> None: ...

def get_max_attestations(
    spec: Any
) -> int: ...

def get_empty_eip7549_aggregation_bits(
    spec: Any,
    state: Any,
    committee_bits: Any,
    slot: int
) -> Bitlist: ...

def get_eip7549_aggregation_bits_offset(
    spec: Any,
    state: Any,
    slot: int,
    committee_bits: Any,
    committee_index: int
) -> int: ...
```