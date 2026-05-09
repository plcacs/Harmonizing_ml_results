from random import Random
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
from eth2spec.test.helpers.voluntary_exits import (
    exit_validators,
    get_exited_validators,
)
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with
from eth2spec.test.helpers.random import (
    randomize_attestation_participation,
    randomize_previous_epoch_participation,
    randomize_state,
)
from eth2spec.test.helpers.rewards import leaking
from eth2spec.typing import spec_types

def run_process_inactivity_updates(spec: spec_types.Spec, state: spec_types.State) -> Iterable[spec_types.EpochProcessingTestOutput]:
    ...

@with_altair_and_later
@spec_state_test
def test_genesis(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_genesis_random_scores(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

def run_inactivity_scores_test(
    spec: spec_types.Spec,
    state: spec_types.State,
    participation_fn: Optional[Callable[[spec_types.Spec, spec_types.State, Random], None]] = None,
    inactivity_scores_fn: Optional[Callable[[spec_types.Spec, spec_types.State, Random], None]] = None,
    rng: Random = Random(10101),
) -> Iterable[spec_types.EpochProcessingTestOutput]:
    ...

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_empty_participation(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_empty_participation_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_random_participation(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_random_participation_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_full_participation(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_full_participation_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_empty_participation(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_empty_participation_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_random_participation(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_random_participation_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_full_participation(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_full_participation_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

def slash_some_validators_for_inactivity_scores_test(
    spec: spec_types.Spec,
    state: spec_types.State,
    rng: Random = Random(40404040),
) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_some_slashed_zero_scores_full_participation(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_slashed_zero_scores_full_participation_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_some_slashed_full_random(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_slashed_full_random_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_exited_full_random_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

def _run_randomized_state_test_for_inactivity_updates(
    spec: spec_types.Spec,
    state: spec_types.State,
    rng: Random = Random(13377331),
) -> Iterable[spec_types.EpochProcessingTestOutput]:
    ...

@with_altair_and_later
@spec_state_test
def test_randomized_state(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...

@with_altair_and_later
@spec_state_test
@leaking()
def test_randomized_state_leaking(spec: spec_types.Spec, state: spec_types.State) -> None:
    ...