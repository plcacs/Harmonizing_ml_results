from typing import Any, Generator, Optional, Tuple, Callable, TypeVar, overload, Union
from random import Random
from lru import LRU
from eth2spec.utils.ssz.ssz_typing import Container, List

# Type variables for the spec and state to maintain consistency across the stub
T_Spec = TypeVar("T_Spec")
T_State = TypeVar("T_State")

class Deltas(Container):
    rewards: List[int]
    penalties: List[int]

def get_inactivity_penalty_quotient(spec: T_Spec) -> int: ...

def has_enough_for_reward(spec: T_Spec, state: T_State, index: int) -> bool: ...

def has_enough_for_leak_penalty(spec: T_Spec, state: T_State, index: int) -> bool: ...

def run_deltas(
    spec: T_Spec, 
    state: T_State
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def deltas_name_to_flag_index(spec: T_Spec, deltas_name: str) -> int: ...

def run_attestation_component_deltas(
    spec: T_Spec,
    state: T_State,
    component_delta_fn: Callable[[T_State], Tuple[List[int], List[int]]],
    matching_att_fn: Callable[[T_State, int], Any],
    deltas_name: str,
) -> Generator[Tuple[str, Deltas], None, None]: ...

def run_get_inclusion_delay_deltas(
    spec: T_Spec, 
    state: T_State
) -> Generator[Tuple[str, Deltas], None, None]: ...

def run_get_inactivity_penalty_deltas(
    spec: T_Spec, 
    state: T_State
) -> Generator[Tuple[str, Deltas], None, None]: ...

def transition_state_to_leak(spec: T_Spec, state: T_State, epochs: Optional[int] = None) -> None: ...

_cache_dict: LRU[Tuple[Any, ...], Any] ...

def leaking(epochs: Optional[int] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

def run_test_empty(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_full_all_correct(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_full_but_partial_participation(
    spec: T_Spec, 
    state: T_State, 
    rng: Random = ...
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_partial(
    spec: T_Spec, 
    state: T_State, 
    fraction_filled: float
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_half_full(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_one_attestation_one_correct(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_with_not_yet_activated_validators(
    spec: T_Spec, 
    state: T_State, 
    rng: Random = ...
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_with_exited_validators(
    spec: T_Spec, 
    state: T_State, 
    rng: Random = ...
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_with_slashed_validators(
    spec: T_Spec, 
    state: T_State, 
    rng: Random = ...
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_some_very_low_effective_balances_that_attested(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_some_very_low_effective_balances_that_did_not_attest(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_full_fraction_incorrect(
    spec: T_Spec, 
    state: T_State, 
    correct_target: bool, 
    correct_head: bool, 
    fraction_incorrect: float
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_full_delay_one_slot(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_full_delay_max_slots(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_full_mixed_delay(
    spec: T_Spec, 
    state: T_State, 
    rng: Random = ...
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_proposer_not_in_attestations(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_duplicate_attestations_at_later_slots(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_all_balances_too_low_for_reward(spec: T_Spec, state: T_State) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...

def run_test_full_random(
    spec: T_Spec, 
    state: T_State, 
    rng: Random = ...
) -> Generator[Tuple[str, Union[T_State, Deltas]], None, None]: ...