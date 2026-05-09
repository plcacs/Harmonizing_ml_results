import random
from typing import Any, Callable, List, Optional, Tuple, Union
from eth2spec.test.context import spec_state_test, spec_test, single_phase, always_bls, with_phases, with_all_phases
from eth2spec.test.helpers.constants import PHASE0
from eth2spec.test.helpers.attestations import Attestation
from eth2spec.test.helpers.block import BeaconBlock
from eth2spec.test.helpers.deposits import Deposit
from eth2spec.test.helpers.keys import Privkey, Pubkey
from eth2spec.test.helpers.state import State
from eth2spec.utils.ssz.ssz_typing import Bitlist
from eth2spec.ethspec import Spec
from eth2spec.ethspec.types import (
    AttestationData,
    BeaconCommittee,
    Domain,
    Epoch,
    Eth1Block,
    Fork,
    ForkDigest,
    Gwei,
    Hash32,
    Root,
    Slot,
    ValidatorIndex,
    Version,
)

def run_get_signature_test(spec: Spec, state: State, obj: Any, domain: Domain, get_signature_fn: Callable[..., bytes], privkey: Privkey, pubkey: Pubkey, signing_ssz_object: Optional[Any] = None) -> None: ...

def run_get_committee_assignment(spec: Spec, state: State, epoch: Epoch, validator_index: ValidatorIndex, valid: bool = True) -> None: ...

def run_is_candidate_block(spec: Spec, eth1_block: Eth1Block, period_start: int, success: bool = True) -> None: ...

def get_min_new_period_epochs(spec: Spec) -> int: ...

def get_mock_aggregate(spec: Spec) -> Attestation: ...

@with_all_phases
@spec_state_test
def test_check_if_validator_active(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_current_epoch(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_next_epoch(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_out_bound_epoch(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_is_proposer(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_epoch_signature(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_is_candidate_block(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_default_vote(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_consensus_vote(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_tie(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_chain_in_past(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_compute_new_state_root(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_block_signature(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_compute_fork_digest(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_attestation_signature_phase0(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
def test_compute_subnet_for_attestation(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_slot_signature(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
@always_bls
def test_is_aggregator(spec: Spec, state: State) -> None: ...

@with_phases([PHASE0])
@spec_state_test
@always_bls
def test_get_aggregate_signature(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof(spec: Spec, state: State) -> None: ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof_signature(spec: Spec, state: State) -> None: ...

def run_compute_subscribed_subnets_arguments(spec: Spec, rng: random.Random = random.Random(1111)) -> None: ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_1(spec: Spec) -> None: ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_2(spec: Spec) -> None: ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_3(spec: Spec) -> None: ...