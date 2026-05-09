import random
from typing import Any, Optional, Tuple, List, Union, Callable, Sequence, Dict
from eth2spec.utils.ssz.ssz_typing import Bitlist
from eth2spec.test.helpers.keys import Pubkey, Privkey
from eth2spec.types import (
    Attestation,
    AttestationData,
    BeaconBlock,
    BeaconCommittee,
    Domain,
    Eth1Block,
    Fork,
    ForkDigest,
    Hash32,
    Slot,
    ValidatorIndex,
    Bitlist,
)

def run_get_signature_test(
    spec: Any,
    state: Any,
    obj: Any,
    domain: Domain,
    get_signature_fn: Callable[[Any, Any, Privkey], bytes],
    privkey: Privkey,
    pubkey: Pubkey,
    signing_ssz_object: Optional[Any] = None,
) -> None:
    ...

def run_get_committee_assignment(
    spec: Any,
    state: Any,
    epoch: int,
    validator_index: int,
    valid: bool = True,
) -> Optional[Tuple[List[int], int, Slot]]:
    ...

def run_is_candidate_block(
    spec: Any,
    eth1_block: Eth1Block,
    period_start: int,
    success: bool = True,
) -> None:
    ...

def get_min_new_period_epochs(spec: Any) -> int:
    ...

def get_mock_aggregate(spec: Any) -> Attestation:
    ...

@with_all_phases
@spec_state_test
def test_check_if_validator_active(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_current_epoch(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_next_epoch(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_out_bound_epoch(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_is_proposer(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_epoch_signature(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_is_candidate_block(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_default_vote(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_consensus_vote(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_tie(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_chain_in_past(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_compute_new_state_root(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_block_signature(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_compute_fork_digest(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_attestation_signature_phase0(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
def test_compute_subnet_for_attestation(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_slot_signature(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_is_aggregator(spec: Any, state: Any) -> None:
    ...

@with_phases([PHASE0])
@spec_state_test
@always_bls
def test_get_aggregate_signature(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof(spec: Any, state: Any) -> None:
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof_signature(spec: Any, state: Any) -> None:
    ...

def run_compute_subscribed_subnets_arguments(
    spec: Any,
    rng: random.Random = random.Random(1111),
) -> None:
    ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_1(spec: Any) -> None:
    ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_2(spec: Any) -> None:
    ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_3(spec: Any) -> None:
    ...