from typing import Any, Optional, Callable, List, Tuple
import random

from eth2spec.test.context import single_phase, spec_state_test, spec_test, always_bls, with_phases, with_all_phases
from eth2spec.test.helpers.constants import PHASE0
from eth2spec.test.helpers.attestations import build_attestation_data, get_valid_attestation
from eth2spec.test.helpers.block import build_empty_block
from eth2spec.test.helpers.deposits import prepare_state_and_deposit
from eth2spec.test.helpers.keys import privkeys, pubkeys
from eth2spec.test.helpers.state import next_epoch
from eth2spec.utils import bls
from eth2spec.utils.ssz.ssz_typing import Bitlist

def run_get_signature_test(
    spec: Any,
    state: Any,
    obj: Any,
    domain: Any,
    get_signature_fn: Callable[[Any, Any, int], Any],
    privkey: int,
    pubkey: bytes,
    signing_ssz_object: Optional[Any] = None
) -> None:
    if signing_ssz_object is None:
        signing_ssz_object = obj
    signature = get_signature_fn(state, obj, privkey)
    signing_root = spec.compute_signing_root(signing_ssz_object, domain)
    assert bls.Verify(pubkey, signing_root, signature)

def run_get_committee_assignment(
    spec: Any,
    state: Any,
    epoch: Any,
    validator_index: Any,
    valid: bool = True
) -> None:
    try:
        assignment: Tuple[Any, Any, Any] = spec.get_committee_assignment(state, epoch, validator_index)
        committee, committee_index, slot = assignment
        assert spec.compute_epoch_at_slot(slot) == epoch
        assert committee == spec.get_beacon_committee(state, slot, committee_index)
        assert committee_index < spec.get_committee_count_per_slot(state, epoch)
        assert validator_index in committee
        assert valid
    except AssertionError:
        assert not valid
    else:
        assert valid

def run_is_candidate_block(
    spec: Any,
    eth1_block: Any,
    period_start: Any,
    success: bool = True
) -> None:
    assert success == spec.is_candidate_block(eth1_block, period_start)

def get_min_new_period_epochs(spec: Any) -> int:
    return spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE * 2 // spec.config.SECONDS_PER_SLOT // spec.SLOTS_PER_EPOCH

def get_mock_aggregate(spec: Any) -> Any:
    return spec.Attestation(data=spec.AttestationData(slot=10))

@with_all_phases
@spec_state_test
def test_check_if_validator_active(spec: Any, state: Any) -> None:
    active_validator_index = len(state.validators) - 1
    assert spec.check_if_validator_active(state, active_validator_index)
    new_validator_index = len(state.validators)
    amount = spec.MAX_EFFECTIVE_BALANCE
    deposit = prepare_state_and_deposit(spec, state, new_validator_index, amount, signed=True)
    spec.process_deposit(state, deposit)
    assert not spec.check_if_validator_active(state, new_validator_index)

@with_all_phases
@spec_state_test
def test_get_committee_assignment_current_epoch(spec: Any, state: Any) -> None:
    epoch = spec.get_current_epoch(state)
    validator_index = len(state.validators) - 1
    run_get_committee_assignment(spec, state, epoch, validator_index, valid=True)

@with_all_phases
@spec_state_test
def test_get_committee_assignment_next_epoch(spec: Any, state: Any) -> None:
    epoch = spec.get_current_epoch(state) + 1
    validator_index = len(state.validators) - 1
    run_get_committee_assignment(spec, state, epoch, validator_index, valid=True)

@with_all_phases
@spec_state_test
def test_get_committee_assignment_out_bound_epoch(spec: Any, state: Any) -> None:
    epoch = spec.get_current_epoch(state) + 2
    validator_index = len(state.validators) - 1
    run_get_committee_assignment(spec, state, epoch, validator_index, valid=False)

@with_all_phases
@spec_state_test
def test_is_proposer(spec: Any, state: Any) -> None:
    proposer_index = spec.get_beacon_proposer_index(state)
    assert spec.is_proposer(state, proposer_index)
    proposer_index = (proposer_index + 1) % len(state.validators)
    assert not spec.is_proposer(state, proposer_index)

@with_all_phases
@spec_state_test
def test_get_epoch_signature(spec: Any, state: Any) -> None:
    block: Any = spec.BeaconBlock()
    privkey: int = privkeys[0]
    pubkey: bytes = pubkeys[0]
    domain = spec.get_domain(state, spec.DOMAIN_RANDAO, spec.compute_epoch_at_slot(block.slot))
    run_get_signature_test(
        spec=spec,
        state=state,
        obj=block,
        domain=domain,
        get_signature_fn=spec.get_epoch_signature,
        privkey=privkey,
        pubkey=pubkey,
        signing_ssz_object=spec.compute_epoch_at_slot(block.slot)
    )

@with_all_phases
@spec_state_test
def test_is_candidate_block(spec: Any, state: Any) -> None:
    distance_duration = spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE
    period_start = distance_duration * 2 + 1000
    run_is_candidate_block(spec, spec.Eth1Block(timestamp=period_start - distance_duration), period_start, success=True)
    run_is_candidate_block(spec, spec.Eth1Block(timestamp=period_start - distance_duration + 1), period_start, success=False)
    run_is_candidate_block(spec, spec.Eth1Block(timestamp=period_start - distance_duration * 2), period_start, success=True)
    run_is_candidate_block(spec, spec.Eth1Block(timestamp=period_start - distance_duration * 2 - 1), period_start, success=False)

@with_all_phases
@spec_state_test
def test_get_eth1_vote_default_vote(spec: Any, state: Any) -> None:
    min_new_period_epochs = get_min_new_period_epochs(spec)
    for _ in range(min_new_period_epochs):
        next_epoch(spec, state)
    state.eth1_data_votes = ()
    eth1_chain: List[Any] = []
    eth1_data = spec.get_eth1_vote(state, eth1_chain)
    assert eth1_data == state.eth1_data

@with_all_phases
@spec_state_test
def test_get_eth1_vote_consensus_vote(spec: Any, state: Any) -> None:
    min_new_period_epochs = get_min_new_period_epochs(spec)
    for _ in range(min_new_period_epochs + 2):
        next_epoch(spec, state)
    period_start = spec.voting_period_start_time(state)
    votes_length = spec.get_current_epoch(state) % spec.EPOCHS_PER_ETH1_VOTING_PERIOD
    assert votes_length >= 3
    state.eth1_data_votes = ()
    block_1 = spec.Eth1Block(
        timestamp=period_start - spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE - 1,
        deposit_count=state.eth1_data.deposit_count,
        deposit_root=b'\x04' * 32
    )
    block_2 = spec.Eth1Block(
        timestamp=period_start - spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE,
        deposit_count=state.eth1_data.deposit_count + 1,
        deposit_root=b'\x05' * 32
    )
    eth1_chain = [block_1, block_2]
    eth1_data_votes: List[Any] = []
    eth1_data_votes.append(spec.get_eth1_data(block_1))
    for _ in range(votes_length - 1):
        eth1_data_votes.append(spec.get_eth1_data(block_2))
    state.eth1_data_votes = eth1_data_votes
    eth1_data = spec.get_eth1_vote(state, eth1_chain)
    assert eth1_data.block_hash == block_2.hash_tree_root()

@with_all_phases
@spec_state_test
def test_get_eth1_vote_tie(spec: Any, state: Any) -> None:
    min_new_period_epochs = get_min_new_period_epochs(spec)
    for _ in range(min_new_period_epochs + 1):
        next_epoch(spec, state)
    period_start = spec.voting_period_start_time(state)
    votes_length = spec.get_current_epoch(state) % spec.EPOCHS_PER_ETH1_VOTING_PERIOD
    assert votes_length > 0 and votes_length % 2 == 0
    state.eth1_data_votes = ()
    block_1 = spec.Eth1Block(
        timestamp=period_start - spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE - 1,
        deposit_count=state.eth1_data.deposit_count,
        deposit_root=b'\x04' * 32
    )
    block_2 = spec.Eth1Block(
        timestamp=period_start - spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE,
        deposit_count=state.eth1_data.deposit_count + 1,
        deposit_root=b'\x05' * 32
    )
    eth1_chain = [block_1, block_2]
    eth1_data_votes: List[Any] = []
    for i in range(votes_length):
        block = block_1 if i % 2 == 0 else block_2
        eth1_data_votes.append(spec.get_eth1_data(block))
    state.eth1_data_votes = eth1_data_votes
    eth1_data = spec.get_eth1_vote(state, eth1_chain)
    assert eth1_data.block_hash == eth1_chain[0].hash_tree_root()

@with_all_phases
@spec_state_test
def test_get_eth1_vote_chain_in_past(spec: Any, state: Any) -> None:
    min_new_period_epochs = get_min_new_period_epochs(spec)
    for _ in range(min_new_period_epochs + 1):
        next_epoch(spec, state)
    period_start = spec.voting_period_start_time(state)
    votes_length = spec.get_current_epoch(state) % spec.EPOCHS_PER_ETH1_VOTING_PERIOD
    assert votes_length > 0 and votes_length % 2 == 0
    state.eth1_data_votes = ()
    block_1 = spec.Eth1Block(
        timestamp=period_start - spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE,
        deposit_count=state.eth1_data.deposit_count - 1,
        deposit_root=b'B' * 32
    )
    eth1_chain = [block_1]
    eth1_data_votes: List[Any] = []
    state.eth1_data_votes = eth1_data_votes
    eth1_data = spec.get_eth1_vote(state, eth1_chain)
    assert eth1_data == state.eth1_data

@with_all_phases
@spec_state_test
def test_compute_new_state_root(spec: Any, state: Any) -> None:
    pre_state: Any = state.copy()
    post_state: Any = state.copy()
    block = build_empty_block(spec, state, state.slot + 1)
    state_root = spec.compute_new_state_root(state, block)
    assert state_root != pre_state.hash_tree_root()
    assert state == pre_state
    spec.process_slots(post_state, block.slot)
    spec.process_block(post_state, block)
    assert state_root == post_state.hash_tree_root()

@with_all_phases
@spec_state_test
@always_bls
def test_get_block_signature(spec: Any, state: Any) -> None:
    privkey: int = privkeys[0]
    pubkey: bytes = pubkeys[0]
    block = build_empty_block(spec, state)
    domain = spec.get_domain(state, spec.DOMAIN_BEACON_PROPOSER, spec.compute_epoch_at_slot(block.slot))
    run_get_signature_test(
        spec=spec,
        state=state,
        obj=block,
        domain=domain,
        get_signature_fn=spec.get_block_signature,
        privkey=privkey,
        pubkey=pubkey
    )

@with_all_phases
@spec_state_test
def test_compute_fork_digest(spec: Any, state: Any) -> None:
    actual_fork_digest = spec.compute_fork_digest(state.fork.current_version, state.genesis_validators_root)
    expected_fork_data_root = spec.hash_tree_root(
        spec.ForkData(current_version=state.fork.current_version, genesis_validators_root=state.genesis_validators_root)
    )
    expected_fork_digest = spec.ForkDigest(expected_fork_data_root[:4])
    assert actual_fork_digest == expected_fork_digest

@with_all_phases
@spec_state_test
@always_bls
def test_get_attestation_signature_phase0(spec: Any, state: Any) -> None:
    privkey: int = privkeys[0]
    pubkey: bytes = pubkeys[0]
    attestation = get_valid_attestation(spec, state, signed=False)
    domain = spec.get_domain(state, spec.DOMAIN_BEACON_ATTESTER, attestation.data.target.epoch)
    run_get_signature_test(
        spec=spec,
        state=state,
        obj=attestation.data,
        domain=domain,
        get_signature_fn=spec.get_attestation_signature,
        privkey=privkey,
        pubkey=pubkey
    )

@with_all_phases
@spec_state_test
def test_compute_subnet_for_attestation(spec: Any, state: Any) -> None:
    for committee_idx in range(spec.MAX_COMMITTEES_PER_SLOT):
        for slot in range(state.slot, state.slot + spec.SLOTS_PER_EPOCH):
            committees_per_slot = spec.get_committee_count_per_slot(state, spec.compute_epoch_at_slot(slot))
            actual_subnet_id = spec.compute_subnet_for_attestation(committees_per_slot, slot, committee_idx)
            slots_since_epoch_start = slot % spec.SLOTS_PER_EPOCH
            committees_since_epoch_start = committees_per_slot * slots_since_epoch_start
            expected_subnet_id = (committees_since_epoch_start + committee_idx) % spec.config.ATTESTATION_SUBNET_COUNT
            assert actual_subnet_id == expected_subnet_id

@with_all_phases
@spec_state_test
@always_bls
def test_get_slot_signature(spec: Any, state: Any) -> None:
    privkey: int = privkeys[0]
    pubkey: bytes = pubkeys[0]
    slot = spec.Slot(10)
    domain = spec.get_domain(state, spec.DOMAIN_SELECTION_PROOF, spec.compute_epoch_at_slot(slot))
    run_get_signature_test(
        spec=spec,
        state=state,
        obj=slot,
        domain=domain,
        get_signature_fn=spec.get_slot_signature,
        privkey=privkey,
        pubkey=pubkey
    )

@with_all_phases
@spec_state_test
@always_bls
def test_is_aggregator(spec: Any, state: Any) -> None:
    slot = state.slot
    committee_index = 0
    has_aggregator = False
    beacon_committee = spec.get_beacon_committee(state, slot, committee_index)
    for validator_index in beacon_committee:
        privkey = privkeys[validator_index]
        slot_signature = spec.get_slot_signature(state, slot, privkey)
        if spec.is_aggregator(state, slot, committee_index, slot_signature):
            has_aggregator = True
            break
    assert has_aggregator

@with_phases([PHASE0])
@spec_state_test
@always_bls
def test_get_aggregate_signature(spec: Any, state: Any) -> None:
    attestations: List[Any] = []
    attesting_pubkeys: List[bytes] = []
    slot = state.slot
    committee_index = 0
    attestation_data = build_attestation_data(spec, state, slot=slot, index=committee_index)
    beacon_committee = spec.get_beacon_committee(state, attestation_data.slot, attestation_data.index)
    committee_size = len(beacon_committee)
    aggregation_bits = Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE](*[0] * committee_size)
    for i, validator_index in enumerate(beacon_committee):
        bits = aggregation_bits.copy()
        bits[i] = True
        attestations.append(
            spec.Attestation(
                data=attestation_data,
                aggregation_bits=bits,
                signature=spec.get_attestation_signature(state, attestation_data, privkeys[validator_index])
            )
        )
        attesting_pubkeys.append(state.validators[validator_index].pubkey)
    assert len(attestations) > 0
    signature = spec.get_aggregate_signature(attestations)
    domain = spec.get_domain(state, spec.DOMAIN_BEACON_ATTESTER, attestation_data.target.epoch)
    signing_root = spec.compute_signing_root(attestation_data, domain)
    assert bls.FastAggregateVerify(attesting_pubkeys, signing_root, signature)

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof(spec: Any, state: Any) -> None:
    privkey: int = privkeys[0]
    aggregator_index = spec.ValidatorIndex(10)
    aggregate = get_mock_aggregate(spec)
    aggregate_and_proof = spec.get_aggregate_and_proof(state, aggregator_index, aggregate, privkey)
    assert aggregate_and_proof.aggregator_index == aggregator_index
    assert aggregate_and_proof.aggregate == aggregate
    assert aggregate_and_proof.selection_proof == spec.get_slot_signature(state, aggregate.data.slot, privkey)

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof_signature(spec: Any, state: Any) -> None:
    privkey: int = privkeys[0]
    pubkey: bytes = pubkeys[0]
    aggregate = get_mock_aggregate(spec)
    aggregate_and_proof = spec.get_aggregate_and_proof(state, spec.ValidatorIndex(1), aggregate, privkey)
    domain = spec.get_domain(state, spec.DOMAIN_AGGREGATE_AND_PROOF, spec.compute_epoch_at_slot(aggregate.data.slot))
    run_get_signature_test(
        spec=spec,
        state=state,
        obj=aggregate_and_proof,
        domain=domain,
        get_signature_fn=spec.get_aggregate_and_proof_signature,
        privkey=privkey,
        pubkey=pubkey
    )

def run_compute_subscribed_subnets_arguments(spec: Any, rng: random.Random = random.Random(1111)) -> None:
    node_id = rng.randint(0, 2 ** 256 - 1)
    epoch = rng.randint(0, 2 ** 64 - 1)
    subnets: List[int] = spec.compute_subscribed_subnets(node_id, epoch)
    assert len(subnets) == spec.config.SUBNETS_PER_NODE

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_1(spec: Any) -> None:
    rng = random.Random(1111)
    run_compute_subscribed_subnets_arguments(spec, rng)

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_2(spec: Any) -> None:
    rng = random.Random(2222)
    run_compute_subscribed_subnets_arguments(spec, rng)

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_3(spec: Any) -> None:
    rng = random.Random(3333)
    run_compute_subscribed_subnets_arguments(spec, rng)