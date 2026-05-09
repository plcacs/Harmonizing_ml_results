from random import Random
from lru import LRU
from eth2spec.phase0.mainnet import VALIDATOR_REGISTRY_LIMIT
from eth2spec.test.helpers.forks import is_post_altair, is_post_bellatrix
from eth2spec.test.helpers.state import next_epoch
from eth2spec.test.helpers.random import (
    set_some_new_deposits,
    exit_random_validators,
    slash_random_validators,
    randomize_state,
)
from eth2spec.test.helpers.attestations import cached_prepare_state_with_attestations
from eth2spec.utils.ssz.ssz_typing import Container, uint64, List
from eth2spec.typing import BeaconState

class Deltas(Container):
    pass

def get_inactivity_penalty_quotient(spec: object) -> int: ...

def has_enough_for_reward(spec: object, state: BeaconState, index: int) -> bool: ...

def has_enough_for_leak_penalty(spec: object, state: BeaconState, index: int) -> bool: ...

def run_deltas(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def deltas_name_to_flag_index(spec: object, deltas_name: str) -> int: ...

def run_attestation_component_deltas(
    spec: object,
    state: BeaconState,
    component_delta_fn: object,
    matching_att_fn: object,
    deltas_name: str
) -> tuple[str, Deltas]: ...

def run_get_inclusion_delay_deltas(spec: object, state: BeaconState) -> tuple[str, Deltas]: ...

def run_get_inactivity_penalty_deltas(spec: object, state: BeaconState) -> tuple[str, Deltas]: ...

def transition_state_to_leak(spec: object, state: BeaconState, epochs: int | None = None) -> None: ...

def leaking(epochs: int | None = None) -> object: ...

def run_test_empty(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_full_all_correct(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_full_but_partial_participation(
    spec: object,
    state: BeaconState,
    rng: Random = Random(5522)
) -> tuple[str, BeaconState]: ...

def run_test_partial(
    spec: object,
    state: BeaconState,
    fraction_filled: float
) -> tuple[str, BeaconState]: ...

def run_test_half_full(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_one_attestation_one_correct(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_with_not_yet_activated_validators(
    spec: object,
    state: BeaconState,
    rng: Random = Random(5555)
) -> tuple[str, BeaconState]: ...

def run_test_with_exited_validators(
    spec: object,
    state: BeaconState,
    rng: Random = Random(1337)
) -> tuple[str, BeaconState]: ...

def run_test_with_slashed_validators(
    spec: object,
    state: BeaconState,
    rng: Random = Random(3322)
) -> tuple[str, BeaconState]: ...

def run_test_some_very_low_effective_balances_that_attested(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_some_very_low_effective_balances_that_did_not_attest(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_full_fraction_incorrect(
    spec: object,
    state: BeaconState,
    correct_target: bool,
    correct_head: bool,
    fraction_incorrect: float
) -> tuple[str, BeaconState]: ...

def run_test_full_delay_one_slot(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_full_delay_max_slots(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_full_mixed_delay(
    spec: object,
    state: BeaconState,
    rng: Random = Random(1234)
) -> tuple[str, BeaconState]: ...

def run_test_proposer_not_in_attestations(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_duplicate_attestations_at_later_slots(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_all_balances_too_low_for_reward(spec: object, state: BeaconState) -> tuple[str, BeaconState]: ...

def run_test_full_random(
    spec: object,
    state: BeaconState,
    rng: Random = Random(8020)
) -> tuple[str, BeaconState]: ...

_cache_dict: LRU = ...