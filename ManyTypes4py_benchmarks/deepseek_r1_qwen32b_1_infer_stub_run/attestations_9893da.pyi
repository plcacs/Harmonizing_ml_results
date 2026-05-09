from lru import LRU
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.state import BeaconState
from eth2spec.test.helpers.block import SignedBlock
from eth2spec.test.helpers.forks import Fork
from eth2spec.test.helpers.keys import Privkey
from eth2spec.utils.bls import BLSSignature
from eth2spec.utils.ssz.ssz_typing import Bitlist
from eth2spec.typing import Attestation, AttestationData, Checkpoint, CommitteeIndex, Epoch, Root, Slot, ValidatorIndex

def run_attestation_processing(spec: Fork, state: BeaconState, attestation: Attestation, valid: bool = True) -> Generator[Tuple[str, Any], None, None]:
    ...

def build_attestation_data(spec: Fork, state: BeaconState, slot: Slot, index: CommitteeIndex, beacon_block_root: Optional[Root] = None, shard: Optional[int] = None) -> AttestationData:
    ...

def get_valid_attestation(spec: Fork, state: BeaconState, slot: Optional[Slot] = None, index: Optional[CommitteeIndex] = None, filter_participant_set: Optional[Callable[[set[ValidatorIndex]], set[ValidatorIndex]]] = None, beacon_block_root: Optional[Root] = None, signed: bool = False) -> Attestation:
    ...

def sign_aggregate_attestation(spec: Fork, state: BeaconState, attestation_data: AttestationData, participants: list[ValidatorIndex]) -> BLSSignature:
    ...

def sign_indexed_attestation(spec: Fork, state: BeaconState, indexed_attestation: Any) -> None:
    ...

def sign_attestation(spec: Fork, state: BeaconState, attestation: Attestation) -> None:
    ...

def get_attestation_signature(spec: Fork, state: BeaconState, attestation_data: AttestationData, privkey: Privkey) -> BLSSignature:
    ...

def compute_max_inclusion_slot(spec: Fork, attestation: Attestation) -> int:
    ...

def fill_aggregate_attestation(spec: Fork, state: BeaconState, attestation: Attestation, committee_index: CommitteeIndex, signed: bool = False, filter_participant_set: Optional[Callable[[set[ValidatorIndex]], set[ValidatorIndex]]] = None) -> None:
    ...

def add_attestations_to_state(spec: Fork, state: BeaconState, attestations: list[Attestation], slot: Slot) -> None:
    ...

def get_valid_attestations_at_slot(state: BeaconState, spec: Fork, slot_to_attest: Slot, participation_fn: Optional[Callable[[Slot, CommitteeIndex, set[ValidatorIndex]], set[ValidatorIndex]]] = None, beacon_block_root: Optional[Root] = None) -> Generator[Attestation, None, None]:
    ...

def get_valid_attestation_at_slot(state: BeaconState, spec: Fork, slot_to_attest: Slot, participation_fn: Optional[Callable[[Slot, CommitteeIndex, set[ValidatorIndex]], set[ValidatorIndex]]] = None, beacon_block_root: Optional[Root] = None) -> Attestation:
    ...

def next_slots_with_attestations(spec: Fork, state: BeaconState, slot_count: int, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[Callable[[Slot, CommitteeIndex, set[ValidatorIndex]], set[ValidatorIndex]]] = None) -> Tuple[BeaconState, list[SignedBlock], BeaconState]:
    ...

def _add_valid_attestations(spec: Fork, state: BeaconState, block: Any, slot_to_attest: Slot, participation_fn: Optional[Callable[[Slot, CommitteeIndex, set[ValidatorIndex]], set[ValidatorIndex]]] = None) -> None:
    ...

def next_epoch_with_attestations(spec: Fork, state: BeaconState, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[Callable[[Slot, CommitteeIndex, set[ValidatorIndex]], set[ValidatorIndex]]] = None) -> Tuple[BeaconState, list[SignedBlock], BeaconState]:
    ...

def state_transition_with_full_block(spec: Fork, state: BeaconState, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[Callable[[Slot, CommitteeIndex, set[ValidatorIndex]], set[ValidatorIndex]]] = None, sync_aggregate: Optional[Any] = None, block: Optional[Any] = None) -> SignedBlock:
    ...

def state_transition_with_full_attestations_block(spec: Fork, state: BeaconState, fill_cur_epoch: bool, fill_prev_epoch: bool) -> SignedBlock:
    ...

def prepare_state_with_attestations(spec: Fork, state: BeaconState, participation_fn: Optional[Callable[[Slot, CommitteeIndex, set[ValidatorIndex]], set[ValidatorIndex]]] = None) -> list[Attestation]:
    ...

def cached_prepare_state_with_attestations(spec: Fork, state: BeaconState) -> None:
    ...

def get_max_attestations(spec: Fork) -> int:
    ...

def get_empty_eip7549_aggregation_bits(spec: Fork, state: BeaconState, committee_bits: Bitlist[ValidatorIndex], slot: Slot) -> Bitlist[ValidatorIndex]:
    ...

def get_eip7549_aggregation_bits_offset(spec: Fork, state: BeaconState, slot: Slot, committee_bits: Bitlist[ValidatorIndex], committee_index: CommitteeIndex) -> int:
    ...

_prep_state_cache_dict: LRU[Tuple[Fork, bytes], bytes] = ...