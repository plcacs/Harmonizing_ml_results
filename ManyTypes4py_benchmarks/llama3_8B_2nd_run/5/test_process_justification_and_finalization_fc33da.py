from typing import Generator, Tuple
from eth2spec.test.context import spec_state_test, with_all_phases
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with
from eth2spec.test.helpers.forks import is_post_altair
from eth2spec.test.helpers.state import transition_to, next_epoch_via_block, next_slot
from eth2spec.test.helpers.voluntary_exits import get_unslashed_exited_validators

def run_process_just_and_fin(spec: object, state: object) -> Generator[None, None, None]:
    yield from run_epoch_processing_with(spec, state, 'process_justification_and_finalization')

def add_mock_attestations(spec: object, state: object, epoch: int, source: object, target: object, sufficient_support: bool = False, messed_up_target: bool = False) -> None:
    # ... (rest of the function remains the same)

def get_checkpoints(spec: object, epoch: int) -> Tuple[object, object, object, object, object]:
    # ... (rest of the function remains the same)

def put_checkpoints_in_block_roots(spec: object, state: object, checkpoints: Tuple[object, ...]) -> None:
    # ... (rest of the function remains the same)

def finalize_on_234(spec: object, state: object, epoch: int, sufficient_support: bool) -> None:
    # ... (rest of the function remains the same)

def finalize_on_23(spec: object, state: object, epoch: int, sufficient_support: bool) -> None:
    # ... (rest of the function remains the same)

def finalize_on_123(spec: object, state: object, epoch: int, sufficient_support: bool) -> None:
    # ... (rest of the function remains the same)

def finalize_on_12(spec: object, state: object, epoch: int, sufficient_support: bool, messed_up_target: bool) -> None:
    # ... (rest of the function remains the same)

@with_all_phases
@spec_state_test
def test_234_ok_support(spec: object, state: object) -> None:
    yield from finalize_on_234(spec, state, 5, True)

@with_all_phases
@spec_state_test
def test_234_poor_support(spec: object, state: object) -> None:
    yield from finalize_on_234(spec, state, 5, False)

@with_all_phases
@spec_state_test
def test_23_ok_support(spec: object, state: object) -> None:
    yield from finalize_on_23(spec, state, 4, True)

@with_all_phases
@spec_state_test
def test_23_poor_support(spec: object, state: object) -> None:
    yield from finalize_on_23(spec, state, 4, False)

@with_all_phases
@spec_state_test
def test_123_ok_support(spec: object, state: object) -> None:
    yield from finalize_on_123(spec, state, 6, True)

@with_all_phases
@spec_state_test
def test_123_poor_support(spec: object, state: object) -> None:
    yield from finalize_on_123(spec, state, 6, False)

@with_all_phases
@spec_state_test
def test_12_ok_support(spec: object, state: object) -> None:
    yield from finalize_on_12(spec, state, 3, True, False)

@with_all_phases
@spec_state_test
def test_12_ok_support_messed_target(spec: object, state: object) -> None:
    yield from finalize_on_12(spec, state, 3, True, True)

@with_all_phases
@spec_state_test
def test_12_poor_support(spec: object, state: object) -> None:
    yield from finalize_on_12(spec, state, 3, False, False)

@with_all_phases
@spec_state_test
def test_balance_threshold_with_exited_validators(spec: object, state: object) -> None:
    # ... (rest of the function remains the same)
