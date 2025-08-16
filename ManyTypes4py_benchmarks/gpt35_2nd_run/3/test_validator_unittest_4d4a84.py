from random import Random
from eth2spec import Spec
from eth2spec.phase0 import (
    Attestation,
    AttestationData,
    BeaconBlock,
    Eth1Block,
    ForkData,
    Slot,
    ValidatorIndex,
    MAX_COMMITTEES_PER_SLOT,
    MAX_EFFECTIVE_BALANCE,
    MAX_VALIDATORS_PER_COMMITTEE,
    SLOTS_PER_EPOCH,
    EPOCHS_PER_ETH1_VOTING_PERIOD,
    ATTESTATION_SUBNET_COUNT,
    DOMAIN_RANDAO,
    DOMAIN_BEACON_PROPOSER,
    DOMAIN_BEACON_ATTESTER,
    DOMAIN_SELECTION_PROOF,
    DOMAIN_AGGREGATE_AND_PROOF,
)
from eth2spec.test.context import single_phase, spec_state_test, spec_test, always_bls, with_phases, with_all_phases
from eth2spec.test.helpers.attestations import build_attestation_data, get_valid_attestation
from eth2spec.test.helpers.block import build_empty_block
from eth2spec.test.helpers.deposits import prepare_state_and_deposit
from eth2spec.test.helpers.keys import privkeys, pubkeys
from eth2spec.test.helpers.state import next_epoch
from eth2spec.utils import bls
from eth2spec.utils.ssz.ssz_typing import Bitlist

def run_get_signature_test(spec: Spec, state, obj, domain, get_signature_fn, privkey, pubkey, signing_ssz_object=None):
    ...

def run_get_committee_assignment(spec: Spec, state, epoch, validator_index, valid=True):
    ...

def run_is_candidate_block(spec: Spec, eth1_block, period_start, success=True):
    ...

def get_min_new_period_epochs(spec: Spec):
    ...

def get_mock_aggregate(spec: Spec):
    ...

@with_all_phases
@spec_state_test
def test_check_if_validator_active(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_current_epoch(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_next_epoch(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_committee_assignment_out_bound_epoch(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_is_proposer(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_epoch_signature(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_is_candidate_block(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_default_vote(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_consensus_vote(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_tie(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_get_eth1_vote_chain_in_past(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_compute_new_state_root(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_block_signature(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_compute_fork_digest(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_attestation_signature_phase0(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
def test_compute_subnet_for_attestation(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_slot_signature(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_is_aggregator(spec: Spec, state):
    ...

@with_phases([PHASE0])
@spec_state_test
@always_bls
def test_get_aggregate_signature(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof(spec: Spec, state):
    ...

@with_all_phases
@spec_state_test
@always_bls
def test_get_aggregate_and_proof_signature(spec: Spec, state):
    ...

def run_compute_subscribed_subnets_arguments(spec: Spec, rng: Random):
    ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_1(spec: Spec):
    ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_2(spec: Spec):
    ...

@with_all_phases
@spec_test
@single_phase
def test_compute_subscribed_subnets_random_3(spec: Spec):
    ...
