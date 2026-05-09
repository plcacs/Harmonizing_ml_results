from random import Random
from lru import LRU
from eth2spec.phase0.mainnet import VALIDATOR_REGISTRY_LIMIT
from eth2spec.test.helpers.forks import is_post_altair, is_post_bellatrix
from eth2spec.test.helpers.state import next_epoch
from eth2spec.test.helpers.random import set_some_new_deposits, exit_random_validators, slash_random_validators, randomize_state
from eth2spec.test.helpers.attestations import cached_prepare_state_with_attestations
from eth2spec.utils.ssz.ssz_typing import Container, uint64, List
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

class Deltas(Container):
    ...

def get_inactivity_penalty_quotient(spec: Any) -> int:
    ...

def has_enough_for_reward(spec: Any, state: Any, index: int) -> bool:
    ...

def has_enough_for_leak_penalty(spec: Any, state: Any, index: int) -> bool:
    ...

def run_deltas(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def deltas_name_to_flag_index(spec: Any, deltas_name: str) -> int:
    ...

def run_attestation_component_deltas(
    spec: Any,
    state: Any,
    component_delta_fn: Callable[[Any], Tuple[List[int], List[int]]],
    matching_att_fn: Callable[[Any, Any], Any],
    deltas_name: str
) -> Generator[Tuple[str, Deltas], None, None]:
    ...

def run_get_inclusion_delay_deltas(spec: Any, state: Any) -> Generator[Tuple[str, Deltas], None, None]:
    ...

def run_get_inactivity_penalty_deltas(spec: Any, state: Any) -> Generator[Tuple[str, Deltas], None, None]:
    ...

def transition_state_to_leak(spec: Any, state: Any, epochs: Optional[int] = None) -> None:
    ...

def leaking(epochs: Optional[int] = None) -> Callable[[Callable], Callable]:
    ...

def run_test_empty(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_full_all_correct(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_full_but_partial_participation(spec: Any, state: Any, rng: Random = ...) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_partial(spec: Any, state: Any, fraction_filled: float) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_half_full(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_one_attestation_one_correct(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_with_not_yet_activated_validators(spec: Any, state: Any, rng: Random = ...) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_with_exited_validators(spec: Any, state: Any, rng: Random = ...) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_with_slashed_validators(spec: Any, state: Any, rng: Random = ...) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_some_very_low_effective_balances_that_attested(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_some_very_low_effective_balances_that_did_not_attest(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_full_fraction_incorrect(
    spec: Any,
    state: Any,
    correct_target: bool,
    correct_head: bool,
    fraction_incorrect: float
) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_full_delay_one_slot(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_full_delay_max_slots(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_full_mixed_delay(spec: Any, state: Any, rng: Random = ...) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_proposer_not_in_attestations(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_duplicate_attestations_at_later_slots(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_all_balances_too_low_for_reward(spec: Any, state: Any) -> Generator[Tuple[str, Any], None, None]:
    ...

def run_test_full_random(spec: Any, state: Any, rng: Random = ...) -> Generator[Tuple[str, Any], None, None]:
    ...

_cache_dict: LRU = ...