from random import Random
from eth2spec import Spec
from eth2spec.test.context import single_phase, spec_state_test, spec_test, always_bls, with_phases, with_all_phases
from eth2spec.test.helpers.constants import PHASE0
from eth2spec.test.helpers.attestations import build_attestation_data, get_valid_attestation
from eth2spec.test.helpers.block import build_empty_block
from eth2spec.test.helpers.deposits import prepare_state_and_deposit
from eth2spec.test.helpers.keys import privkeys, pubkeys
from eth2spec.test.helpers.state import next_epoch
from eth2spec.utils import bls
from eth2spec.utils.ssz.ssz_typing import Bitlist

def run_get_signature_test(spec: Spec, state, obj, domain, get_signature_fn, privkey, pubkey, signing_ssz_object=None):
    if signing_ssz_object is None:
        signing_ssz_object = obj
    signature = get_signature_fn(state, obj, privkey)
    signing_root = spec.compute_signing_root(signing_ssz_object, domain)
    assert bls.Verify(pubkey, signing_root, signature)

def run_get_committee_assignment(spec: Spec, state, epoch, validator_index, valid=True):
    try:
        assignment = spec.get_committee_assignment(state, epoch, validator_index)
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

def run_is_candidate_block(spec: Spec, eth1_block, period_start, success=True):
    assert success == spec.is_candidate_block(eth1_block, period_start)

def get_min_new_period_epochs(spec: Spec):
    return spec.config.SECONDS_PER_ETH1_BLOCK * spec.config.ETH1_FOLLOW_DISTANCE * 2 // spec.config.SECONDS_PER_SLOT // spec.SLOTS_PER_EPOCH

def get_mock_aggregate(spec: Spec):
    return spec.Attestation(data=spec.AttestationData(slot=10))

def run_compute_subscribed_subnets_arguments(spec: Spec, rng: Random):
    node_id = rng.randint(0, 2 ** 256 - 1)
    epoch = rng.randint(0, 2 ** 64 - 1)
    subnets = spec.compute_subscribed_subnets(node_id, epoch)
    assert len(subnets) == spec.config.SUBNETS_PER_NODE
