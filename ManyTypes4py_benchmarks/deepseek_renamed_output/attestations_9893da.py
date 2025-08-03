from lru import LRU
from typing import List, Set, Callable, Optional, Generator, Tuple, Dict, Any, TypeVar
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.state import state_transition_and_sign_block, next_epoch, next_slot
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.forks import is_post_altair, is_post_deneb, is_post_electra
from eth2spec.test.helpers.keys import privkeys
from eth2spec.utils import bls
from eth2spec.utils.ssz.ssz_typing import Bitlist
from eth2spec.phase0 import spec as phase0_spec
from eth2spec.altair import spec as altair_spec
from eth2spec.bellatrix import spec as bellatrix_spec
from eth2spec.capella import spec as capella_spec
from eth2spec.deneb import spec as deneb_spec
from eth2spec.electra import spec as electra_spec

T = TypeVar('T')
SpecType = TypeVar('SpecType', phase0_spec, altair_spec, bellatrix_spec, capella_spec, deneb_spec, electra_spec)
StateType = TypeVar('StateType')
AttestationType = TypeVar('AttestationType')
BlockType = TypeVar('BlockType')
IndexedAttestationType = TypeVar('IndexedAttestationType')
SignedBlockType = TypeVar('SignedBlockType')
ParticipationFn = Callable[[int, int, Set[int]], Set[int]]
ParticipantsFilter = Callable[[Set[int]], Set[int]]

def func_w2xqv8rk(spec: SpecType, state: StateType, attestation: AttestationType, valid: bool = True) -> Generator[Tuple[str, Any], None, None]:
    yield 'pre', state
    yield 'attestation', attestation
    if not valid:
        expect_assertion_error(lambda: spec.process_attestation(state, attestation))
        yield 'post', None
        return
    if not is_post_altair(spec):
        current_epoch_count = len(state.current_epoch_attestations)
        previous_epoch_count = len(state.previous_epoch_attestations)
    spec.process_attestation(state, attestation)
    if not is_post_altair(spec):
        if attestation.data.target.epoch == spec.get_current_epoch(state):
            assert len(state.current_epoch_attestations) == current_epoch_count + 1
        else:
            assert len(state.previous_epoch_attestations) == previous_epoch_count + 1
    else:
        pass
    yield 'post', state

def func_1z7cr9f7(spec: SpecType, state: StateType, slot: int, index: int, beacon_block_root: Optional[bytes] = None, shard: Optional[int] = None) -> Any:
    assert state.slot >= slot
    if beacon_block_root is not None:
        pass
    elif slot == state.slot:
        beacon_block_root = build_empty_block_for_next_slot(spec, state).parent_root
    else:
        beacon_block_root = spec.get_block_root_at_slot(state, slot)
    current_epoch_start_slot = spec.compute_start_slot_at_epoch(spec.get_current_epoch(state))
    if slot < current_epoch_start_slot:
        epoch_boundary_root = spec.get_block_root(state, spec.get_previous_epoch(state))
    elif slot == current_epoch_start_slot:
        epoch_boundary_root = beacon_block_root
    else:
        epoch_boundary_root = spec.get_block_root(state, spec.get_current_epoch(state))
    if slot < current_epoch_start_slot:
        source_epoch = state.previous_justified_checkpoint.epoch
        source_root = state.previous_justified_checkpoint.root
    else:
        source_epoch = state.current_justified_checkpoint.epoch
        source_root = state.current_justified_checkpoint.root
    data = spec.AttestationData(
        slot=slot,
        index=0 if is_post_electra(spec) else index,
        beacon_block_root=beacon_block_root,
        source=spec.Checkpoint(epoch=source_epoch, root=source_root),
        target=spec.Checkpoint(epoch=spec.compute_epoch_at_slot(slot), root=epoch_boundary_root)
    )
    return data

def func_zteoga80(spec: SpecType, state: StateType, slot: Optional[int] = None, index: Optional[int] = None, filter_participant_set: Optional[ParticipantsFilter] = None, beacon_block_root: Optional[bytes] = None, signed: bool = False) -> AttestationType:
    if slot is None:
        slot = state.slot
    if index is None:
        index = 0
    attestation_data = func_1z7cr9f7(spec, state, slot=slot, index=index, beacon_block_root=beacon_block_root)
    attestation = spec.Attestation(data=attestation_data)
    func_g7e276et(spec, state, attestation, index, signed=signed, filter_participant_set=filter_participant_set)
    return attestation

def func_tj36j7rw(spec: SpecType, state: StateType, attestation_data: Any, participants: List[int]) -> bytes:
    signatures = []
    for validator_index in participants:
        privkey = privkeys[validator_index]
        signatures.append(func_nm9agqxc(spec, state, attestation_data, privkey))
    return bls.Aggregate(signatures)

def func_1g30ilcf(spec: SpecType, state: StateType, indexed_attestation: IndexedAttestationType) -> None:
    participants = indexed_attestation.attesting_indices
    data = indexed_attestation.data
    indexed_attestation.signature = func_tj36j7rw(spec, state, data, participants)

def func_0l0u0prm(spec: SpecType, state: StateType, attestation: AttestationType) -> None:
    participants = spec.get_attesting_indices(state, attestation)
    attestation.signature = func_tj36j7rw(spec, state, attestation.data, participants)

def func_nm9agqxc(spec: SpecType, state: StateType, attestation_data: Any, privkey: bytes) -> bytes:
    domain = spec.get_domain(state, spec.DOMAIN_BEACON_ATTESTER, attestation_data.target.epoch)
    signing_root = spec.compute_signing_root(attestation_data, domain)
    return bls.Sign(privkey, signing_root)

def func_ax2vj0fm(spec: SpecType, attestation: AttestationType) -> int:
    if is_post_deneb(spec):
        next_epoch = spec.compute_epoch_at_slot(attestation.data.slot) + 1
        end_of_next_epoch = spec.compute_start_slot_at_epoch(next_epoch + 1) - 1
        return end_of_next_epoch
    return attestation.data.slot + spec.SLOTS_PER_EPOCH

def func_g7e276et(spec: SpecType, state: StateType, attestation: AttestationType, committee_index: int, signed: bool = False, filter_participant_set: Optional[ParticipantsFilter] = None) -> None:
    beacon_committee = spec.get_beacon_committee(state, attestation.data.slot, committee_index)
    participants = set(beacon_committee)
    if filter_participant_set is not None:
        participants = filter_participant_set(participants)
    if is_post_electra(spec):
        attestation.committee_bits[committee_index] = True
        attestation.aggregation_bits = func_91v1eqf4(spec, state, attestation.committee_bits, attestation.data.slot)
    else:
        committee_size = len(beacon_committee)
        attestation.aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*([0] * committee_size))
    for i in range(len(beacon_committee)):
        if is_post_electra(spec):
            offset = func_zinsedn8(spec, state, attestation.data.slot, attestation.committee_bits, committee_index)
            aggregation_bits_index = offset + i
            attestation.aggregation_bits[aggregation_bits_index] = beacon_committee[i] in participants
        else:
            attestation.aggregation_bits[i] = beacon_committee[i] in participants
    if signed and len(participants) > 0:
        func_0l0u0prm(spec, state, attestation)

def func_gnddphni(spec: SpecType, state: StateType, attestations: List[AttestationType], slot: int) -> None:
    if state.slot < slot:
        spec.process_slots(state, slot)
    for attestation in attestations:
        spec.process_attestation(state, attestation)

def func_hj76ehjo(state: StateType, spec: SpecType, slot_to_attest: int, participation_fn: Optional[ParticipationFn] = None, beacon_block_root: Optional[bytes] = None) -> Generator[AttestationType, None, None]:
    committees_per_slot = spec.get_committee_count_per_slot(state, spec.compute_epoch_at_slot(slot_to_attest))
    for index in range(committees_per_slot):
        def participants_filter(comm: Set[int]) -> Set[int]:
            if participation_fn is None:
                return comm
            else:
                return participation_fn(state.slot, index, comm)
        yield func_zteoga80(spec, state, slot_to_attest, index=index, signed=True, filter_participant_set=participants_filter, beacon_block_root=beacon_block_root)

def func_ig8pct6a(state: StateType, spec: SpecType, slot_to_attest: int, participation_fn: Optional[ParticipationFn] = None, beacon_block_root: Optional[bytes] = None) -> AttestationType:
    assert is_post_electra(spec)
    attestations = list(func_hj76ehjo(state, spec, slot_to_attest, participation_fn=participation_fn, beacon_block_root=beacon_block_root))
    if not attestations:
        raise Exception('No valid attestations found')
    return spec.compute_on_chain_aggregate(attestations)

def func_yc8ar4lm(spec: SpecType, state: StateType, slot_count: int, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[ParticipationFn] = None) -> Tuple[StateType, List[SignedBlockType], StateType]:
    post_state = state.copy()
    signed_blocks = []
    for _ in range(slot_count):
        signed_block = func_81tcrik7(spec, post_state, fill_cur_epoch, fill_prev_epoch, participation_fn)
        signed_blocks.append(signed_block)
    return state, signed_blocks, post_state

def func_04a3fe0n(spec: SpecType, state: StateType, block: BlockType, slot_to_attest: int, participation_fn: Optional[ParticipationFn] = None) -> None:
    if is_post_electra(spec):
        attestation = func_ig8pct6a(state, spec, slot_to_attest, participation_fn=participation_fn)
        block.body.attestations.append(attestation)
    else:
        attestations = func_hj76ehjo(state, spec, slot_to_attest, participation_fn=participation_fn)
        for attestation in attestations:
            block.body.attestations.append(attestation)

def func_tufxqzz9(spec: SpecType, state: StateType, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[ParticipationFn] = None) -> Tuple[StateType, List[SignedBlockType], StateType]:
    assert state.slot % spec.SLOTS_PER_EPOCH == 0
    return func_yc8ar4lm(spec, state, spec.SLOTS_PER_EPOCH, fill_cur_epoch, fill_prev_epoch, participation_fn)

def func_81tcrik7(spec: SpecType, state: StateType, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[ParticipationFn] = None, sync_aggregate: Optional[Any] = None, block: Optional[BlockType] = None) -> SignedBlockType:
    if block is None:
        block = build_empty_block_for_next_slot(spec, state)
    if fill_cur_epoch and state.slot >= spec.MIN_ATTESTATION_INCLUSION_DELAY:
        slot_to_attest = state.slot - spec.MIN_ATTESTATION_INCLUSION_DELAY + 1
        if slot_to_attest >= spec.compute_start_slot_at_epoch(spec.get_current_epoch(state)):
            func_04a3fe0n(spec, state, block, slot_to_attest, participation_fn=participation_fn)
    if fill_prev_epoch and state.slot >= spec.SLOTS_PER_EPOCH:
        slot_to_attest = state.slot - spec.SLOTS_PER_EPOCH + 1
        func_04a3fe0n(spec, state, block, slot_to_attest, participation_fn=participation_fn)
    if sync_aggregate is not None:
        block.body.sync_aggregate = sync_aggregate
    signed_block = state_transition_and_sign_block(spec, state, block)
    return signed_block

def func_zo4tbrcb(spec: SpecType, state: StateType, fill_cur_epoch: bool, fill_prev_epoch: bool) -> SignedBlockType:
    block = build_empty_block_for_next_slot(spec, state)
    attestations = []
    if fill_cur_epoch:
        slots = state.slot % spec.SLOTS_PER_EPOCH
        for slot_offset in range(slots):
            target_slot = state.slot - slot_offset
            attestations += list(func_hj76ehjo(state, spec, target_slot))
    if fill_prev_epoch:
        slots = spec.SLOTS_PER_EPOCH - state.slot % spec.SLOTS_PER_EPOCH
        for slot_offset in range(1, slots):
            target_slot = (state.slot - state.slot % spec.SLOTS_PER_EPOCH - slot_offset)
            attestations += list(func_hj76ehjo(state, spec, target_slot))
    block.body.attestations = attestations
    signed_block = state_transition_and_sign_block(spec, state, block)
    return signed_block

def func_vcr2w0mp(spec: SpecType, state: StateType, participation_fn: Optional[ParticipationFn] = None) -> List[AttestationType]:
    next_epoch(spec, state)
    start_slot = state.slot
    start_epoch = spec.get_current_epoch(state)
    next_epoch_start_slot = spec.compute_start_slot_at_epoch(start_epoch + 1)
    attestations = []
    for _ in range(spec.SLOTS_PER_EPOCH + spec.MIN_ATTESTATION_INCLUSION_DELAY):
        if state.slot < next_epoch_start_slot:
            for committee_index in range(spec.get_committee_count_per_slot(state, spec.get_current_epoch(state))):
                def temp_participants_filter(comm: Set[int]) -> Set[int]:
                    if participation_fn is None:
                        return comm
                    else:
                        return participation_fn(state.slot, committee_index, comm)
                attestation = func_zteoga80(spec, state, index=committee_index, filter_participant_set=temp_participants_filter, signed=True)
                if any(attestation.aggregation_bits):
                    attestations.append(attestation)
        if state.slot >= start_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY:
            inclusion_slot = state.slot - spec.MIN_ATTESTATION_INCLUSION_DELAY
            include_attestations = [att for att in attestations if att.data.slot == inclusion_slot]
            func_gnddphni(spec, state, include_attestations, state.slot)
        next_slot(spec, state)
    assert state.slot == next_epoch_start_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY
    if not is_post_altair(spec):
        assert len(state.previous_epoch_attestations) == len(attestations)
    return attestations

_prep_state_cache_dict: Dict[Tuple[str, bytes], Any] = LRU(size=10)

def func_emn7aedw(spec: SpecType, state: StateType) -> None:
    key = spec.fork, state.hash_tree_root()
    global _prep_state_cache_dict
    if key not in _prep_state_cache_dict:
        func_vcr2w0mp(spec, state)
        _prep_state_cache_dict[key] = state.get_backing()
    state.set_backing(_prep_state_cache_dict[key])

def func_y3s87jj6(spec: SpecType) -> int:
    if is_post_electra(spec):
        return spec.MAX_ATTESTATIONS_ELECTRA
    else:
        return spec.MAX_ATTESTATIONS

def func_91v1eqf4(spec: SpecType, state: StateType, committee_bits: Any, slot: int) -> Bitlist:
    committee_indices = spec.get_committee_indices(committee_bits)
    participants_count = 0
    for index in committee_indices:
        committee = spec.get_beacon_committee(state, slot, index)
        participants_count += len(committee)
    aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE * spec.MAX_COMMITTEES_PER_SLOT]([False]