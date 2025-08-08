from random import Random
from lru import LRU
from eth2spec.phase0.mainnet import VALIDATOR_REGISTRY_LIMIT
from eth2spec.test.helpers.forks import is_post_altair, is_post_bellatrix
from eth2spec.test.helpers.state import next_epoch
from eth2spec.test.helpers.random import set_some_new_deposits, exit_random_validators, slash_random_validators, randomize_state
from eth2spec.test.helpers.attestations import cached_prepare_state_with_attestations
from eth2spec.utils.ssz.ssz_typing import Container, uint64, List

class Deltas(Container):
    rewards: List[uint64]
    penalties: List[uint64]

def get_inactivity_penalty_quotient(spec) -> uint64:
    if is_post_bellatrix(spec):
        return spec.INACTIVITY_PENALTY_QUOTIENT_BELLATRIX
    elif is_post_altair(spec):
        return spec.INACTIVITY_PENALTY_QUOTIENT_ALTAIR
    else:
        return spec.INACTIVITY_PENALTY_QUOTIENT

def has_enough_for_reward(spec, state, index) -> bool:
    ...

def has_enough_for_leak_penalty(spec, state, index) -> bool:
    ...

def run_deltas(spec, state):
    ...

def deltas_name_to_flag_index(spec, deltas_name) -> uint64:
    ...

def run_attestation_component_deltas(spec, state, component_delta_fn, matching_att_fn, deltas_name):
    ...

def run_get_inclusion_delay_deltas(spec, state):
    ...

def run_get_inactivity_penalty_deltas(spec, state):
    ...

def transition_state_to_leak(spec, state, epochs=None):
    ...

_cache_dict: LRU = LRU(size=10)

def leaking(epochs=None):

    def deco(fn):

        def entry(*args, spec, state, **kw):
            ...

def run_test_empty(spec, state):
    ...

def run_test_full_all_correct(spec, state):
    ...

def run_test_full_but_partial_participation(spec, state, rng=Random(5522)):
    ...

def run_test_partial(spec, state, fraction_filled) -> float:
    ...

def run_test_half_full(spec, state):
    ...

def run_test_one_attestation_one_correct(spec, state):
    ...

def run_test_with_not_yet_activated_validators(spec, state, rng=Random(5555)):
    ...

def run_test_with_exited_validators(spec, state, rng=Random(1337)):
    ...

def run_test_with_slashed_validators(spec, state, rng=Random(3322)):
    ...

def run_test_some_very_low_effective_balances_that_attested(spec, state):
    ...

def run_test_some_very_low_effective_balances_that_did_not_attest(spec, state):
    ...

def run_test_full_fraction_incorrect(spec, state, correct_target, correct_head, fraction_incorrect) -> float:
    ...

def run_test_full_delay_one_slot(spec, state):
    ...

def run_test_full_delay_max_slots(spec, state):
    ...

def run_test_full_mixed_delay(spec, state, rng=Random(1234)):
    ...

def run_test_proposer_not_in_attestations(spec, state):
    ...

def run_test_duplicate_attestations_at_later_slots(spec, state):
    ...

def run_test_all_balances_too_low_for_reward(spec, state):
    ...

def run_test_full_random(spec, state, rng=Random(8020)):
    ...
