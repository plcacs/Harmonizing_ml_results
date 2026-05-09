from random import Random
from lru import LRU
from eth2spec.phase0.mainnet import Spec
from eth2spec.test.helpers.forks import is_post_altair, is_post_bellatrix
from eth2spec.test.helpers.state import State
from eth2spec.utils.ssz.ssz_typing import Container, uint64, List
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

class Deltas(Container):
    rewards: List[uint64]
    penalties: List[uint64]

def get_inactivity_penalty_quotient(spec: Spec) -> int:
    ...

def has_enough_for_reward(spec: Spec, state: State, index: int) -> bool:
    ...

def has_enough_for_leak_penalty(spec: Spec, state: State, index: int) -> bool:
    ...

def run_deltas(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def deltas_name_to_flag_index(spec: Spec, deltas_name: str) -> int:
    ...

def run_attestation_component_deltas(
    spec: Spec,
    state: State,
    component_delta_fn: Callable[[State], Tuple[List[uint64], List[uint64]]],
    matching_att_fn: Callable[[State, int], Any],
    deltas_name: str
) -> Generator[Tuple[str, Deltas], None, None]:
    ...

def run_get_inclusion_delay_deltas(spec: Spec, state: State) -> Generator[Tuple[str, Deltas], None, None]:
    ...

def run_get_inactivity_penalty_deltas(spec: Spec, state: State) -> Generator[Tuple[str, Deltas], None, None]:
    ...

def transition_state_to_leak(spec: Spec, state: State, epochs: Optional[int] = None) -> None:
    ...

def leaking(epochs: Optional[int] = None) -> Callable[[Callable], Callable]:
    ...

def run_test_empty(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_full_all_correct(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_full_but_partial_participation(spec: Spec, state: State, rng: Random = Random(5522)) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_partial(spec: Spec, state: State, fraction_filled: float) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_half_full(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_one_attestation_one_correct(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_with_not_yet_activated_validators(spec: Spec, state: State, rng: Random = Random(5555)) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_with_exited_validators(spec: Spec, state: State, rng: Random = Random(1337)) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_with_slashed_validators(spec: Spec, state: State, rng: Random = Random(3322)) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_some_very_low_effective_balances_that_attested(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_some_very_low_effective_balances_that_did_not_attest(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_full_fraction_incorrect(
    spec: Spec,
    state: State,
    correct_target: bool,
    correct_head: bool,
    fraction_incorrect: float
) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_full_delay_one_slot(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_full_delay_max_slots(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_full_mixed_delay(spec: Spec, state: State, rng: Random = Random(1234)) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_proposer_not_in_attestations(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_duplicate_attestations_at_later_slots(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_all_balances_too_low_for_reward(spec: Spec, state: State) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...

def run_test_full_random(spec: Spec, state: State, rng: Random = Random(8020)) -> Generator[Union[Tuple[str, State], Tuple[str, Deltas]], None, None]:
    ...