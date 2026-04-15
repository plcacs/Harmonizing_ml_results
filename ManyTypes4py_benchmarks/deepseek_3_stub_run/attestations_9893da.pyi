from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from lru import LRU
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.state import state_transition_and_sign_block, next_epoch, next_slot
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.forks import is_post_altair, is_post_deneb, is_post_electra
from eth2spec.test.helpers.keys import privkeys
from eth2spec.utils import bls
from eth2spec.utils.ssz.ssz_typing import Bitlist

T = TypeVar('T')
SpecType = TypeVar('SpecType')
StateType = TypeVar('StateType')
AttestationType = TypeVar('AttestationType')
BlockType = TypeVar('BlockType')
SignedBlockType = TypeVar('SignedBlockType')
CheckpointType = TypeVar('CheckpointType')
AttestationDataType = TypeVar('AttestationDataType')
IndexedAttestationType = TypeVar('IndexedAttestationType')
SyncAggregateType = TypeVar('SyncAggregateType')
CommitteeBitsType = TypeVar('CommitteeBitsType')

def run_attestation_processing(
    spec: Any,
    state: StateType,
    attestation: AttestationType,
    valid: bool = True
) -> Generator[Tuple[str, Optional[StateType]], None, None]: ...

def build_attestation_data(
    spec: Any,
    state: StateType,
    slot: int,
    index: int,
    beacon_block_root: Optional[bytes] = None,
    shard: Optional[int] = None
) -> AttestationDataType: ...

def get_valid_attestation(
    spec: Any,
    state: StateType,
    slot: Optional[int] = None,
    index: Optional[int] = None,
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = None,
    beacon_block_root: Optional[bytes] = None,
    signed: bool = False
) -> AttestationType: ...

def sign_aggregate_attestation(
    spec: Any,
    state: StateType,
    attestation_data: AttestationDataType,
    participants: List[int]
) -> bytes: ...

def sign_indexed_attestation(
    spec: Any,
    state: StateType,
    indexed_attestation: IndexedAttestationType
) -> None: ...

def sign_attestation(
    spec: Any,
    state: StateType,
    attestation: AttestationType
) -> None: ...

def get_attestation_signature(
    spec: Any,
    state: StateType,
    attestation_data: AttestationDataType,
    privkey: bytes
) -> bytes: ...

def compute_max_inclusion_slot(
    spec: Any,
    attestation: AttestationType
) -> int: ...

def fill_aggregate_attestation(
    spec: Any,
    state: StateType,
    attestation: AttestationType,
    committee_index: int,
    signed: bool = False,
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = None
) -> None: ...

def add_attestations_to_state(
    spec: Any,
    state: StateType,
    attestations: List[AttestationType],
    slot: int
) -> None: ...

def get_valid_attestations_at_slot(
    state: StateType,
    spec: Any,
    slot_to_attest: int,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None,
    beacon_block_root: Optional[bytes] = None
) -> Iterator[AttestationType]: ...

def get_valid_attestation_at_slot(
    state: StateType,
    spec: Any,
    slot_to_attest: int,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None,
    beacon_block_root: Optional[bytes] = None
) -> AttestationType: ...

def next_slots_with_attestations(
    spec: Any,
    state: StateType,
    slot_count: int,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> Tuple[StateType, List[SignedBlockType], StateType]: ...

def _add_valid_attestations(
    spec: Any,
    state: StateType,
    block: BlockType,
    slot_to_attest: int,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> None: ...

def next_epoch_with_attestations(
    spec: Any,
    state: StateType,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> Tuple[StateType, List[SignedBlockType], StateType]: ...

def state_transition_with_full_block(
    spec: Any,
    state: StateType,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None,
    sync_aggregate: Optional[SyncAggregateType] = None,
    block: Optional[BlockType] = None
) -> SignedBlockType: ...

def state_transition_with_full_attestations_block(
    spec: Any,
    state: StateType,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool
) -> SignedBlockType: ...

def prepare_state_with_attestations(
    spec: Any,
    state: StateType,
    participation_fn: Optional[Callable[[int, int, Set[int]], Set[int]]] = None
) -> List[AttestationType]: ...

_prep_state_cache_dict: LRU[Tuple[str, bytes], Any] = ...

def cached_prepare_state_with_attestations(
    spec: Any,
    state: StateType
) -> None: ...

def get_max_attestations(spec: Any) -> int: ...

def get_empty_eip7549_aggregation_bits(
    spec: Any,
    state: StateType,
    committee_bits: CommitteeBitsType,
    slot: int
) -> Bitlist: ...

def get_eip7549_aggregation_bits_offset(
    spec: Any,
    state: StateType,
    slot: int,
    committee_bits: CommitteeBitsType,
    committee_index: int
) -> int: ...