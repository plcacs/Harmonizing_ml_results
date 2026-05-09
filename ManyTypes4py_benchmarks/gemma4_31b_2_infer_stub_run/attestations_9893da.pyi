from typing import Any, Generator, Iterable, List, Optional, Set, Tuple, Callable, Union
from eth2spec.utils.ssz.ssz_typing import Bitlist

# Assuming the existence of these types based on the usage in the module
# These would typically be imported from the spec or state modules
# Since they are dynamic (spec.Attestation, etc.), we use Any or generic types
# but try to be as descriptive as possible.

# Type aliases for clarity
Spec = Any
State = Any
Attestation = Any
AttestationData = Any
Checkpoint = Any
Block = Any

def run_attestation_processing(
    spec: Spec, 
    state: State, 
    attestation: Attestation, 
    valid: bool = True
) -> Generator[Tuple[str, Optional[Union[State, Attestation]]], None, None]: ...

def build_attestation_data(
    spec: Spec, 
    state: State, 
    slot: int, 
    index: int, 
    beacon_block_root: Optional[bytes] = None, 
    shard: Optional[int] = None
) -> AttestationData: ...

def get_valid_attestation(
    spec: Spec, 
    state: State, 
    slot: Optional[int] = None, 
    index: Optional[int] = None, 
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = None, 
    beacon_block_root: Optional[bytes] = None, 
    signed: bool = False
) -> Attestation: ...

def sign_aggregate_attestation(
    spec: Spec, 
    state: State, 
    attestation_data: AttestationData, 
    participants: Iterable[int]
) -> Any: ...

def sign_indexed_attestation(spec: Spec, state: State, indexed_attestation: Attestation) -> None: ...

def sign_attestation(spec: Spec, state: State, attestation: Attestation) -> None: ...

def get_attestation_signature(spec: Spec, state: State, attestation_data: AttestationData, privkey: bytes) -> Any: ...

def compute_max_inclusion_slot(spec: Spec, attestation: Attestation) -> int: ...

def fill_aggregate_attestation(
    spec: Spec, 
    state: State, 
    attestation: Attestation, 
    committee_index: int, 
    signed: bool = False, 
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = None
) -> None: ...

def add_attestations_to_state(spec: Spec, state: State, attestations: Iterable[Attestation], slot: int) -> None: ...

def get_valid_attestations_at_slot(
    state: State, 
    spec: Spec, 
    slot_to_attest: int, 
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None, 
    beacon_block_root: Optional[bytes] = None
) -> Generator[Attestation, None, None]: ...

def get_valid_attestation_at_slot(
    state: State, 
    spec: Spec, 
    slot_to_attest: int, 
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None, 
    beacon_block_root: Optional[bytes] = None
) -> Attestation: ...

def next_slots_with_attestations(
    spec: Spec, 
    state: State, 
    slot_count: int, 
    fill_cur_epoch: bool, 
    fill_prev_epoch: bool, 
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> Tuple[State, List[Any], State]: ...

def _add_valid_attestations(
    spec: Spec, 
    state: State, 
    block: Block, 
    slot_to_attest: int, 
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> None: ...

def next_epoch_with_attestations(
    spec: Spec, 
    state: State, 
    fill_cur_epoch: bool, 
    fill_prev_epoch: bool, 
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> Tuple[State, List[Any], State]: ...

def state_transition_with_full_block(
    spec: Spec, 
    state: State, 
    fill_cur_epoch: bool, 
    fill_prev_epoch: bool, 
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None, 
    sync_aggregate: Optional[Any] = None, 
    block: Optional[Block] = None
) -> Any: ...

def state_transition_with_full_attestations_block(
    spec: Spec, 
    state: State, 
    fill_cur_epoch: bool, 
    fill_prev_epoch: bool
) -> Any: ...

def prepare_state_with_attestations(
    spec: Spec, 
    state: State, 
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> List[Attestation]: ...

_prep_state_cache_dict: Any = ...

def cached_prepare_state_with_attestations(spec: Spec, state: State) -> None: ...

def get_max_attestations(spec: Spec) -> int: ...

def get_empty_eip7549_aggregation_bits(spec: Spec, state: State, committee_bits: Any, slot: int) -> Bitlist: ...

def get_eip7549_aggregation_bits_offset(
    spec: Spec, 
    state: State, 
    slot: int, 
    committee_bits: Any, 
    committee_index: int
) -> int: ...