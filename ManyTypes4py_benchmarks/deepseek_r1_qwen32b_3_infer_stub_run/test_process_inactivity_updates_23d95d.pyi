from random import Random
from typing import Any, Generator, Optional
from eth2spec.test.context import spec_state_test
from eth2spec.test.helpers.inactivity_scores import (
    randomize_inactivity_scores,
    zero_inactivity_scores,
)
from eth2spec.test.helpers.state import (
    next_epoch,
    next_epoch_via_block,
    set_full_participation,
    set_empty_participation,
)
from eth2spec.test.helpers.voluntary_exits import get_exited_validators
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with
from eth2spec.test.helpers.random import randomize_attestation_participation
from eth2spec.test.helpers.rewards import leaking


def run_process_inactivity_updates(spec: Any, state: Any) -> Generator[Any, Any, None]: ...


@spec_state_test
def test_genesis(spec: Any, state: Any) -> None: ...


def run_inactivity_scores_test(
    spec: Any,
    state: Any,
    participation_fn: Optional[Any] = None,
    inactivity_scores_fn: Optional[Any] = None,
    rng: Random = Random(10101),
) -> Generator[Any, Any, None]: ...


@spec_state_test
def test_all_zero_inactivity_scores_empty_participation(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_empty_participation_leaking(spec: Any, state: Any) -> None: ...


@spec_state_test
def test_all_zero_inactivity_scores_random_participation(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_random_participation_leaking(spec: Any, state: Any) -> None: ...


@spec_state_test
def test_all_zero_inactivity_scores_full_participation(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_full_participation_leaking(spec: Any, state: Any) -> None: ...


@spec_state_test
def test_random_inactivity_scores_empty_participation(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_random_inactivity_scores_empty_participation_leaking(spec: Any, state: Any) -> None: ...


@spec_state_test
def test_random_inactivity_scores_random_participation(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_random_inactivity_scores_random_participation_leaking(spec: Any, state: Any) -> None: ...


@spec_state_test
def test_random_inactivity_scores_full_participation(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_random_inactivity_scores_full_participation_leaking(spec: Any, state: Any) -> None: ...


def slash_some_validators_for_inactivity_scores_test(
    spec: Any,
    state: Any,
    rng: Random = Random(40404040),
) -> None: ...


@spec_state_test
def test_some_slashed_zero_scores_full_participation(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_some_slashed_zero_scores_full_participation_leaking(spec: Any, state: Any) -> None: ...


@spec_state_test
def test_some_slashed_full_random(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_some_slashed_full_random_leaking(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_some_exited_full_random_leaking(spec: Any, state: Any) -> None: ...


def _run_randomized_state_test_for_inactivity_updates(
    spec: Any,
    state: Any,
    rng: Random = Random(13377331),
) -> Generator[Any, Any, None]: ...


@spec_state_test
def test_randomized_state(spec: Any, state: Any) -> None: ...


@spec_state_test
@leaking()
def test_randomized_state_leaking(spec: Any, state: Any) -> None: ...