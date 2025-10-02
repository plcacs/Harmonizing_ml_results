import random
from collections import defaultdict
from typing import Any, List, Set, Tuple, DefaultDict
from eth2spec.utils.ssz.ssz_typing import Bitvector
from eth2spec.utils import bls
from eth2spec.test.helpers.block import build_empty_block
from eth2spec.test.helpers.keys import pubkey_to_privkey, privkeys, pubkeys
from eth2spec.test.helpers.state import transition_to
from eth2spec.test.helpers.sync_committee import compute_sync_committee_signature
from eth2spec.test.context import always_bls, spec_state_test, with_altair_and_later, with_presets
from eth2spec.test.helpers.constants import MAINNET, MINIMAL

rng: random.Random = random.Random(1337)

def ensure_assignments_in_sync_committee(spec: Any, state: Any, epoch: int, sync_committee: Any, active_pubkeys: List[bytes]) -> None:
    assert len(sync_committee.pubkeys) >= 3
    some_pubkeys: List[bytes] = rng.sample(sync_committee.pubkeys, 3)
    for pubkey in some_pubkeys:
        validator_index: int = active_pubkeys.index(pubkey)
        assert spec.is_assigned_to_sync_committee(state, epoch, validator_index)

@with_altair_and_later
@spec_state_test
def test_is_assigned_to_sync_committee(spec: Any, state: Any) -> None:
    epoch: int = spec.get_current_epoch(state)
    validator_indices: List[int] = spec.get_active_validator_indices(state, epoch)
    validator_count: int = len(validator_indices)
    query_epoch: int = epoch + 1
    next_query_epoch: int = query_epoch + spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD
    active_pubkeys: List[bytes] = [state.validators[index].pubkey for index in validator_indices]
    ensure_assignments_in_sync_committee(spec, state, query_epoch, state.current_sync_committee, active_pubkeys)
    ensure_assignments_in_sync_committee(spec, state, next_query_epoch, state.next_sync_committee, active_pubkeys)
    sync_committee_pubkeys: Set[bytes] = set(list(state.current_sync_committee.pubkeys) + list(state.next_sync_committee.pubkeys))
    disqualified_pubkeys: Set[bytes] = set(filter(lambda key: key not in sync_committee_pubkeys, active_pubkeys))
    if disqualified_pubkeys:
        sample_size: int = 3
        assert validator_count >= sample_size
        some_pubkeys: List[bytes] = rng.sample(sorted(disqualified_pubkeys), sample_size)
        for pubkey in some_pubkeys:
            validator_index: int = active_pubkeys.index(pubkey)
            is_current: bool = spec.is_assigned_to_sync_committee(state, query_epoch, validator_index)
            is_next: bool = spec.is_assigned_to_sync_committee(state, next_query_epoch, validator_index)
            is_current_or_next: bool = is_current or is_next
            assert not is_current_or_next

def _get_sync_committee_signature(spec: Any,
                                  state: Any,
                                  target_slot: int,
                                  target_block_root: bytes,
                                  subcommittee_index: int,
                                  index_in_subcommittee: int) -> bytes:
    subcommittee_size: int = spec.SYNC_COMMITTEE_SIZE // spec.SYNC_COMMITTEE_SUBNET_COUNT
    sync_committee_index: int = subcommittee_index * subcommittee_size + index_in_subcommittee
    pubkey: bytes = state.current_sync_committee.pubkeys[sync_committee_index]
    privkey: int = pubkey_to_privkey[pubkey]
    return compute_sync_committee_signature(spec, state, target_slot, privkey, block_root=target_block_root)

@with_altair_and_later
@with_presets([MINIMAL], reason='too slow')
@spec_state_test
@always_bls
def test_process_sync_committee_contributions(spec: Any, state: Any) -> None:
    transition_to(spec, state, state.slot + 3)
    block: Any = build_empty_block(spec, state)
    previous_slot: int = state.slot - 1
    target_block_root: bytes = spec.get_block_root_at_slot(state, previous_slot)
    aggregation_bits: Bitvector = Bitvector[spec.SYNC_COMMITTEE_SIZE // spec.SYNC_COMMITTEE_SUBNET_COUNT]()
    aggregation_index: int = 0
    aggregation_bits[aggregation_index] = True
    contributions: List[Any] = [
        spec.SyncCommitteeContribution(
            slot=block.slot,
            beacon_block_root=target_block_root,
            subcommittee_index=i,
            aggregation_bits=aggregation_bits,
            signature=_get_sync_committee_signature(spec, state, previous_slot, target_block_root, i, aggregation_index)
        )
        for i in range(spec.SYNC_COMMITTEE_SUBNET_COUNT)
    ]
    empty_sync_aggregate: Any = spec.SyncAggregate()
    empty_sync_aggregate.sync_committee_signature = spec.G2_POINT_AT_INFINITY
    assert block.body.sync_aggregate == empty_sync_aggregate
    spec.process_sync_committee_contributions(block, set(contributions))
    assert len(block.body.sync_aggregate.sync_committee_bits) != 0
    assert block.body.sync_committee.aggregate_signature != spec.G2_POINT_AT_INFINITY or block.body.sync_aggregate.sync_committee_signature != spec.G2_POINT_AT_INFINITY  # Adjusting field name in comparison if needed.
    spec.process_block(state, block)

@with_altair_and_later
@spec_state_test
@always_bls
def test_get_sync_committee_message(spec: Any, state: Any) -> None:
    validator_index: int = 0
    block_root: bytes = spec.Root(b'\x12' * 32)
    sync_committee_message: Any = spec.get_sync_committee_message(state=state, block_root=block_root, validator_index=validator_index, privkey=privkeys[validator_index])
    assert sync_committee_message.slot == state.slot
    assert sync_committee_message.beacon_block_root == block_root
    assert sync_committee_message.validator_index == validator_index
    epoch: int = spec.get_current_epoch(state)
    domain: bytes = spec.get_domain(state, spec.DOMAIN_SYNC_COMMITTEE, epoch)
    signing_root: bytes = spec.compute_signing_root(block_root, domain)
    signature: bytes = bls.Sign(privkeys[validator_index], signing_root)
    assert sync_committee_message.signature == signature

def _validator_index_for_pubkey(state: Any, pubkey: bytes) -> int:
    return list(map(lambda v: v.pubkey, state.validators)).index(pubkey)

def _subnet_for_sync_committee_index(spec: Any, i: int) -> int:
    return i // (spec.SYNC_COMMITTEE_SIZE // spec.SYNC_COMMITTEE_SUBNET_COUNT)

def _get_expected_subnets_by_pubkey(sync_committee_members: List[Tuple[int, bytes]]) -> DefaultDict[bytes, Set[int]]:
    expected_subnets_by_pubkey: DefaultDict[bytes, Set[int]] = defaultdict(set)
    for subnet, pubkey in sync_committee_members:
        expected_subnets_by_pubkey[pubkey].add(subnet)
    return expected_subnets_by_pubkey

@with_altair_and_later
@with_presets([MINIMAL], reason='too slow')
@spec_state_test
def test_compute_subnets_for_sync_committee(state: Any, spec: Any) -> None:
    transition_to(spec, state, spec.SLOTS_PER_EPOCH * spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD)
    next_slot_epoch: int = spec.compute_epoch_at_slot(state.slot + 1)
    assert spec.compute_sync_committee_period(spec.get_current_epoch(state)) == spec.compute_sync_committee_period(next_slot_epoch)
    some_sync_committee_members: List[Tuple[int, bytes]] = [
        (_subnet_for_sync_committee_index(spec, i), pubkey)
        for i, pubkey in enumerate(state.current_sync_committee.pubkeys)
    ]
    expected_subnets_by_pubkey: DefaultDict[bytes, Set[int]] = _get_expected_subnets_by_pubkey(some_sync_committee_members)
    for _, pubkey in some_sync_committee_members:
        validator_index: int = _validator_index_for_pubkey(state, pubkey)
        subnets: Set[int] = spec.compute_subnets_for_sync_committee(state, validator_index)
        expected_subnets: Set[int] = expected_subnets_by_pubkey[pubkey]
        assert subnets == expected_subnets

@with_altair_and_later
@with_presets([MINIMAL], reason='too slow')
@spec_state_test
def test_compute_subnets_for_sync_committee_slot_period_boundary(state: Any, spec: Any) -> None:
    transition_to(spec, state, spec.SLOTS_PER_EPOCH * spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD - 1)
    next_slot_epoch: int = spec.compute_epoch_at_slot(state.slot + 1)
    assert spec.compute_sync_committee_period(spec.get_current_epoch(state)) != spec.compute_sync_committee_period(next_slot_epoch)
    some_sync_committee_members: List[Tuple[int, bytes]] = [
        (_subnet_for_sync_committee_index(spec, i), pubkey)
        for i, pubkey in enumerate(state.next_sync_committee.pubkeys)
    ]
    expected_subnets_by_pubkey: DefaultDict[bytes, Set[int]] = _get_expected_subnets_by_pubkey(some_sync_committee_members)
    for _, pubkey in some_sync_committee_members:
        validator_index: int = _validator_index_for_pubkey(state, pubkey)
        subnets: Set[int] = spec.compute_subnets_for_sync_committee(state, validator_index)
        expected_subnets: Set[int] = expected_subnets_by_pubkey[pubkey]
        assert subnets == expected_subnets

@with_altair_and_later
@spec_state_test
@always_bls
def test_get_sync_committee_selection_proof(spec: Any, state: Any) -> None:
    slot: int = 1
    subcommittee_index: int = 0
    privkey: int = privkeys[1]
    sync_committee_selection_proof: bytes = spec.get_sync_committee_selection_proof(state, slot, subcommittee_index, privkey)
    domain: bytes = spec.get_domain(state, spec.DOMAIN_SYNC_COMMITTEE_SELECTION_PROOF, spec.compute_epoch_at_slot(slot))
    signing_data: Any = spec.SyncAggregatorSelectionData(slot=slot, subcommittee_index=subcommittee_index)
    signing_root: bytes = spec.compute_signing_root(signing_data, domain)
    pubkey: bytes = pubkeys[1]
    assert bls.Verify(pubkey, signing_root, sync_committee_selection_proof)

@with_altair_and_later
@spec_state_test
@with_presets([MAINNET], reason='to test against the mainnet SYNC_COMMITTEE_SIZE')
def test_is_sync_committee_aggregator(spec: Any, state: Any) -> None:
    sample_count: int = int(spec.SYNC_COMMITTEE_SIZE // spec.SYNC_COMMITTEE_SUBNET_COUNT) * 100
    is_aggregator_count: int = 0
    for i in range(sample_count):
        signature: bytes = spec.hash(i.to_bytes(32, byteorder='little'))
        if spec.is_sync_committee_aggregator(signature):
            is_aggregator_count += 1
    lower_bound: float = spec.TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE * 100 * 0.9
    upper_bound: float = spec.TARGET_AGGREGATORS_PER_SYNC_SUBCOMMITTEE * 100 * 1.1
    assert lower_bound <= is_aggregator_count <= upper_bound

@with_altair_and_later
@spec_state_test
def test_get_contribution_and_proof(spec: Any, state: Any) -> None:
    aggregator_index: int = 10
    privkey: int = privkeys[3]
    contribution: Any = spec.SyncCommitteeContribution(
        slot=10,
        beacon_block_root=b'\x12' * 32,
        subcommittee_index=1,
        aggregation_bits=spec.Bitvector[spec.SYNC_COMMITTEE_SIZE // spec.SYNC_COMMITTEE_SUBNET_COUNT](),
        signature=b'2' * 96
    )
    selection_proof: bytes = spec.get_sync_committee_selection_proof(state, contribution.slot, contribution.subcommittee_index, privkey)
    contribution_and_proof: Any = spec.get_contribution_and_proof(state, aggregator_index, contribution, privkey)
    assert contribution_and_proof == spec.ContributionAndProof(
        aggregator_index=aggregator_index,
        contribution=contribution,
        selection_proof=selection_proof
    )

@with_altair_and_later
@spec_state_test
@always_bls
def test_get_contribution_and_proof_signature(spec: Any, state: Any) -> None:
    privkey: int = privkeys[3]
    pubkey: bytes = pubkeys[3]
    contribution_and_proof: Any = spec.ContributionAndProof(
        aggregator_index=10,
        contribution=spec.SyncCommitteeContribution(
            slot=10,
            beacon_block_root=b'\x12' * 32,
            subcommittee_index=1,
            aggregation_bits=spec.Bitvector[spec.SYNC_COMMITTEE_SIZE // spec.SYNC_COMMITTEE_SUBNET_COUNT](),
            signature=b'4' * 96
        ),
        selection_proof=b'V' * 96
    )
    contribution_and_proof_signature: bytes = spec.get_contribution_and_proof_signature(state, contribution_and_proof, privkey)
    contribution: Any = contribution_and_proof.contribution
    domain: bytes = spec.get_domain(state, spec.DOMAIN_CONTRIBUTION_AND_PROOF, spec.compute_epoch_at_slot(contribution.slot))
    signing_root: bytes = spec.compute_signing_root(contribution_and_proof, domain)
    assert bls.Verify(pubkey, signing_root, contribution_and_proof_signature)