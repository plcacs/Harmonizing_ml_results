from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from random import Random
from eth2spec.test.context import spec_state_test, with_altair_and_later
from eth2spec.test.helpers.rewards import leaking

def run_process_inactivity_updates(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_genesis(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_genesis_random_scores(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def run_inactivity_scores_test(
    spec: Any,
    state: Any,
    participation_fn: Optional[Callable[..., Any]] = None,
    inactivity_scores_fn: Optional[Callable[..., Any]] = None,
    rng: Random = ...,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_empty_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_empty_participation_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_random_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_random_participation_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_full_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_full_participation_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_empty_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_empty_participation_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_random_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_random_participation_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_full_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_full_participation_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def slash_some_validators_for_inactivity_scores_test(
    spec: Any,
    state: Any,
    rng: Random = ...,
) -> None: ...

@with_altair_and_later
@spec_state_test
def test_some_slashed_zero_scores_full_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_slashed_zero_scores_full_participation_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_some_slashed_full_random(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_slashed_full_random_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_exited_full_random_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def _run_randomized_state_test_for_inactivity_updates(
    spec: Any,
    state: Any,
    rng: Random = ...,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
def test_randomized_state(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_randomized_state_leaking(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...