from lru import LRU
from typing import List, Callable, Generator, Tuple, Optional, Set, Any
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.state import (
    state_transition_and_sign_block,
    next_epoch,
    next_slot,
)
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.forks import is_post_altair, is_post_deneb, is_post_electra
from eth2spec.test.helpers.keys import privkeys
from eth2spec.utils import bls
from eth2spec.utils.ssz.ssz_typing import Bitlist

# Assuming the types Spec, State, Attestation, SignedBlock, etc. exist
# These would need to be imported from the appropriate eth2spec modules

def run_attestation_processing(
    spec: Any,
    state: Any,
    attestation: Any,
    valid: bool = True,
) -> Generator[Tuple[str, Optional[Any]], None, None]:
    """
    Run ``process_attestation``, yielding:
      - pre-state ('pre')
      - attestation ('attestation')
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    yield ('pre', state)
    yield ('attestation', attestation)
    if not valid:
        expect_assertion_error(lambda: spec.process_attestation(state, attestation))
        yield ('post', None)
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
    yield ('post', state)

def build_attestation_data(
    spec: Any,
    state: Any,
    slot: int,
    index: int,
    beacon_block_root: Optional[Any] = None,
    shard: Optional[Any] = None,
) -> Any:
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
        target=spec.Checkpoint(
            epoch=spec.compute_epoch_at_slot(slot),
            root=epoch_boundary_root
        )
    )
    return data

def get_valid_attestation(
    spec: Any,
    state: Any,
    slot: Optional[int] = None,
    index: Optional[int] = None,
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = None,
    beacon_block_root: Optional[Any] = None,
    signed: bool = False,
) -> Any:
    """
    Return a valid attestation at `slot` and committee index `index`.

    If filter_participant_set filters everything, the attestation has 0 participants, and cannot be signed.
    Thus strictly speaking invalid when no participant is added later.
    """
    if slot is None:
        slot = state.slot
    if index is None:
        index = 0
    attestation_data = build_attestation_data(
        spec,
        state,
        slot=slot,
        index=index,
        beacon_block_root=beacon_block_root
    )
    attestation = spec.Attestation(data=attestation_data)
    fill_aggregate_attestation(
        spec,
        state,
        attestation,
        signed=signed,
        filter_participant_set=filter_participant_set,
        committee_index=index
    )
    return attestation

def sign_aggregate_attestation(
    spec: Any,
    state: Any,
    attestation_data: Any,
    participants: Set[int],
) -> Any:
    signatures = []
    for validator_index in participants:
        privkey = privkeys[validator_index]
        signatures.append(get_attestation_signature(spec, state, attestation_data, privkey))
    return bls.Aggregate(signatures)

def sign_indexed_attestation(
    spec: Any,
    state: Any,
    indexed_attestation: Any,
) -> None:
    participants = indexed_attestation.attesting_indices
    data = indexed_attestation.data
    indexed_attestation.signature = sign_aggregate_attestation(
        spec,
        state,
        data,
        participants
    )

def sign_attestation(
    spec: Any,
    state: Any,
    attestation: Any,
) -> None:
    participants = spec.get_attesting_indices(state, attestation)
    attestation.signature = sign_aggregate_attestation(
        spec,
        state,
        attestation.data,
        participants
    )

def get_attestation_signature(
    spec: Any,
    state: Any,
    attestation_data: Any,
    privkey: Any,
) -> Any:
    domain = spec.get_domain(
        state,
        spec.DOMAIN_BEACON_ATTESTER,
        attestation_data.target.epoch
    )
    signing_root = spec.compute_signing_root(attestation_data, domain)
    return bls.Sign(privkey, signing_root)

def compute_max_inclusion_slot(
    spec: Any,
    attestation: Any,
) -> int:
    if is_post_deneb(spec):
        next_epoch = spec.compute_epoch_at_slot(attestation.data.slot) + 1
        end_of_next_epoch = spec.compute_start_slot_at_epoch(next_epoch + 1) - 1
        return end_of_next_epoch
    return attestation.data.slot + spec.SLOTS_PER_EPOCH

def fill_aggregate_attestation(
    spec: Any,
    state: Any,
    attestation: Any,
    committee_index: int,
    signed: bool = False,
    filter_participant_set: Optional[Callable[[Set[int]], Set[int]]] = None,
) -> None:
    """
     `signed`: Signing is optional.
     `filter_participant_set`: Optional, filters the full committee indices set (default) to a subset that participates
    """
    beacon_committee = spec.get_beacon_committee(state, attestation.data.slot, committee_index)
    participants: Set[int] = set(beacon_committee)
    if filter_participant_set is not None:
        participants = filter_participant_set(participants)
    if is_post_electra(spec):
        attestation.committee_bits[committee_index] = True
        attestation.aggregation_bits = get_empty_eip7549_aggregation_bits(
            spec,
            state,
            attestation.committee_bits,
            attestation.data.slot
        )
    else:
        committee_size = len(beacon_committee)
        attestation.aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*[0] * committee_size)
    for i in range(len(beacon_committee)):
        if is_post_electra(spec):
            offset = get_eip7549_aggregation_bits_offset(
                spec,
                state,
                attestation.data.slot,
                attestation.committee_bits,
                committee_index
            )
            aggregation_bits_index = offset + i
            attestation.aggregation_bits[aggregation_bits_index] = beacon_committee[i] in participants
        else:
            attestation.aggregation_bits[i] = beacon_committee[i] in participants
    if signed and len(participants) > 0:
        sign_attestation(spec, state, attestation)

def add_attestations_to_state(
    spec: Any,
    state: Any,
    attestations: List[Any],
    slot: int,
) -> None:
    if state.slot < slot:
        spec.process_slots(state, slot)
    for attestation in attestations:
        spec.process_attestation(state, attestation)

def get_valid_attestations_at_slot(
    state: Any,
    spec: Any,
    slot_to_attest: int,
    participation_fn: Optional[
        Callable[[int, int, Set[int]], Set[int]]
    ] = None,
    beacon_block_root: Optional[Any] = None,
) -> Generator[Any, None, None]:
    """
    Return attestations at slot `slot_to_attest`.
    """
    committees_per_slot = spec.get_committee_count_per_slot(
        state,
        spec.compute_epoch_at_slot(slot_to_attest)
    )
    for index in range(committees_per_slot):

        def participants_filter(comm: Set[int]) -> Set[int]:
            if participation_fn is None:
                return comm
            else:
                return participation_fn(state.slot, index, comm)

        yield get_valid_attestation(
            spec,
            state,
            slot_to_attest,
            index=index,
            signed=True,
            filter_participant_set=participants_filter,
            beacon_block_root=beacon_block_root
        )

def get_valid_attestation_at_slot(
    state: Any,
    spec: Any,
    slot_to_attest: int,
    participation_fn: Optional[
        Callable[[int, int, Set[int]], Set[int]]
    ] = None,
    beacon_block_root: Optional[Any] = None,
) -> Any:
    """
    Return the aggregate attestation post Electra.
    Note: this EIP supports dense packing of on-chain aggregates so we can just return a single `Attestation`.
    """
    assert is_post_electra(spec)
    attestations = list(
        get_valid_attestations_at_slot(
            state,
            spec,
            slot_to_attest,
            participation_fn=participation_fn,
            beacon_block_root=beacon_block_root
        )
    )
    if not attestations:
        raise Exception('No valid attestations found')
    return spec.compute_on_chain_aggregate(attestations)

def next_slots_with_attestations(
    spec: Any,
    state: Any,
    slot_count: int,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[
        Callable[[int, int, Set[int]], Set[int]]
    ] = None,
) -> Tuple[Any, List[Any], Any]:
    """
    participation_fn: (slot, committee_index, committee_indices_set) -> participants_indices_set
    """
    post_state = state.copy()
    signed_blocks: List[Any] = []
    for _ in range(slot_count):
        signed_block = state_transition_with_full_block(
            spec,
            post_state,
            fill_cur_epoch,
            fill_prev_epoch,
            participation_fn
        )
        signed_blocks.append(signed_block)
    return (state, signed_blocks, post_state)

def _add_valid_attestations(
    spec: Any,
    state: Any,
    block: Any,
    slot_to_attest: int,
    participation_fn: Optional[
        Callable[[int, int, Set[int]], Set[int]]
    ] = None,
) -> None:
    if is_post_electra(spec):
        attestation = get_valid_attestation_at_slot(
            state,
            spec,
            slot_to_attest,
            participation_fn=participation_fn
        )
        block.body.attestations.append(attestation)
    else:
        attestations = get_valid_attestations_at_slot(
            state,
            spec,
            slot_to_attest,
            participation_fn=participation_fn
        )
        for attestation in attestations:
            block.body.attestations.append(attestation)

def next_epoch_with_attestations(
    spec: Any,
    state: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[
        Callable[[int, int, Set[int]], Set[int]]
    ] = None,
) -> Tuple[Any, List[Any], Any]:
    assert state.slot % spec.SLOTS_PER_EPOCH == 0
    return next_slots_with_attestations(
        spec,
        state,
        spec.SLOTS_PER_EPOCH,
        fill_cur_epoch,
        fill_prev_epoch,
        participation_fn
    )

def state_transition_with_full_block(
    spec: Any,
    state: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[
        Callable[[int, int, Set[int]], Set[int]]
    ] = None,
    sync_aggregate: Optional[Any] = None,
    block: Optional[Any] = None,
) -> Any:
    """
    Build and apply a block with attestations at the calculated `slot_to_attest` of current epoch and/or previous epoch.
    """
    if block is None:
        block = build_empty_block_for_next_slot(spec, state)
    if fill_cur_epoch and state.slot >= spec.MIN_ATTESTATION_INCLUSION_DELAY:
        slot_to_attest = state.slot - spec.MIN_ATTESTATION_INCLUSION_DELAY + 1
        if slot_to_attest >= spec.compute_start_slot_at_epoch(spec.get_current_epoch(state)):
            _add_valid_attestations(
                spec,
                state,
                block,
                slot_to_attest,
                participation_fn=participation_fn
            )
    if fill_prev_epoch and state.slot >= spec.SLOTS_PER_EPOCH:
        slot_to_attest = state.slot - spec.SLOTS_PER_EPOCH + 1
        _add_valid_attestations(
            spec,
            state,
            block,
            slot_to_attest,
            participation_fn=participation_fn
        )
    if sync_aggregate is not None:
        block.body.sync_aggregate = sync_aggregate
    signed_block = state_transition_and_sign_block(spec, state, block)
    return signed_block

def state_transition_with_full_attestations_block(
    spec: Any,
    state: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
) -> Any:
    """
    Build and apply a block with attestations at all valid slots of current epoch and/or previous epoch.
    """
    block = build_empty_block_for_next_slot(spec, state)
    attestations: List[Any] = []
    if fill_cur_epoch:
        slots = state.slot % spec.SLOTS_PER_EPOCH
        for slot_offset in range(slots):
            target_slot = state.slot - slot_offset
            attestations += list(
                get_valid_attestations_at_slot(
                    state,
                    spec,
                    target_slot
                )
            )
    if fill_prev_epoch:
        slots = spec.SLOTS_PER_EPOCH - (state.slot % spec.SLOTS_PER_EPOCH)
        for slot_offset in range(1, slots):
            target_slot = state.slot - (state.slot % spec.SLOTS_PER_EPOCH) - slot_offset
            attestations += list(
                get_valid_attestations_at_slot(
                    state,
                    spec,
                    target_slot
                )
            )
    block.body.attestations = attestations
    signed_block = state_transition_and_sign_block(spec, state, block)
    return signed_block

def prepare_state_with_attestations(
    spec: Any,
    state: Any,
    participation_fn: Optional[
        Callable[[int, int, Set[int]], Set[int]]
    ] = None,
) -> List[Any]:
    """
    Prepare state with attestations according to the ``participation_fn``.
    If no ``participation_fn``, default to "full" -- max committee participation at each slot.

    participation_fn: (slot, committee_index, committee_indices_set) -> participants_indices_set
    """
    next_epoch(spec, state)
    start_slot = state.slot
    start_epoch = spec.get_current_epoch(state)
    next_epoch_start_slot = spec.compute_start_slot_at_epoch(start_epoch + 1)
    attestations: List[Any] = []
    for _ in range(spec.SLOTS_PER_EPOCH + spec.MIN_ATTESTATION_INCLUSION_DELAY):
        if state.slot < next_epoch_start_slot:
            for committee_index in range(
                spec.get_committee_count_per_slot(state, spec.get_current_epoch(state))
            ):

                def temp_participants_filter(comm: Set[int]) -> Set[int]:
                    if participation_fn is None:
                        return comm
                    else:
                        return participation_fn(state.slot, committee_index, comm)

                attestation = get_valid_attestation(
                    spec,
                    state,
                    index=committee_index,
                    filter_participant_set=temp_participants_filter,
                    signed=True
                )
                if any(attestation.aggregation_bits):
                    attestations.append(attestation)
        if state.slot >= start_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY:
            inclusion_slot = state.slot - spec.MIN_ATTESTATION_INCLUSION_DELAY
            include_attestations = [
                att for att in attestations if att.data.slot == inclusion_slot
            ]
            add_attestations_to_state(spec, state, include_attestations, state.slot)
        next_slot(spec, state)
    assert state.slot == next_epoch_start_slot + spec.MIN_ATTESTATION_INCLUSION_DELAY
    if not is_post_altair(spec):
        assert len(state.previous_epoch_attestations) == len(attestations)
    return attestations

_prep_state_cache_dict: LRU = LRU(size=10)

def cached_prepare_state_with_attestations(
    spec: Any,
    state: Any,
) -> None:
    """
    Cached version of prepare_state_with_attestations,
    but does not return anything, and does not support a participation fn argument
    """
    key = (spec.fork, state.hash_tree_root())
    global _prep_state_cache_dict
    if key not in _prep_state_cache_dict:
        prepare_state_with_attestations(spec, state)
        _prep_state_cache_dict[key] = state.get_backing()
    state.set_backing(_prep_state_cache_dict[key])

def get_max_attestations(
    spec: Any
) -> int:
    if is_post_electra(spec):
        return spec.MAX_ATTESTATIONS_ELECTRA
    else:
        return spec.MAX_ATTESTATIONS

def get_empty_eip7549_aggregation_bits(
    spec: Any,
    state: Any,
    committee_bits: Any,
    slot: int,
) -> Bitlist:
    committee_indices = spec.get_committee_indices(committee_bits)
    participants_count = 0
    for index in committee_indices:
        committee = spec.get_beacon_committee(state, slot, index)
        participants_count += len(committee)
    aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE * spec.MAX_COMMITTEES_PER_SLOT](
        *([False] * participants_count)
    )
    return aggregation_bits

def get_eip7549_aggregation_bits_offset(
    spec: Any,
    state: Any,
    slot: int,
    committee_bits: Any,
    committee_index: int,
) -> int:
    """
    Calculate the offset for the aggregation bits based on the committee index.
    """
    committee_indices = spec.get_committee_indices(committee_bits)
    assert committee_index in committee_indices
    offset = 0
    for i in committee_indices:
        if committee_index == i:
            break
        committee = spec.get_beacon_committee(state, slot, i)
        offset += len(committee)
    return offset
