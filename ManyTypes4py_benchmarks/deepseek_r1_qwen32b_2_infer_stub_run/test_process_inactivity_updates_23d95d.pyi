from random import Random
from typing import Any, Callable, Generator, Optional
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


def run_process_inactivity_updates(spec: Any, state: Any) -> Generator[Any, Any, None]:
    ...


def run_inactivity_scores_test(
    spec: Any,
    state: Any,
    participation_fn: Optional[Callable[[Any, Any, Any], None]] = None,
    inactivity_scores_fn: Optional[Callable[[Any, Any, Any], None]] = None,
    rng: Random = Random(10101),
) -> Generator[Any, Any, None]:
    ...


def slash_some_validators_for_inactivity_scores_test(
    spec: Any, state: Any, rng: Random = Random(40404040)
) -> None:
    ...


def _run_randomized_state_test_for_inactivity_updates(
    spec: Any, state: Any, rng: Random = Random(13377331)
) -> Generator[Any, Any, None]:
    ...