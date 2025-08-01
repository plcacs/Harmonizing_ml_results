from random import Random
from typing import Iterator, List, Callable, Any, Tuple

from eth2spec.test.helpers.constants import MINIMAL
from eth2spec.test.context import with_altair_and_later, with_custom_state, spec_test, spec_state_test, with_presets, single_phase
from eth2spec.test.helpers.state import next_epoch_via_block
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with

def get_full_flags(spec: Any) -> int:
    full_flags = spec.ParticipationFlags(0)
    for flag_index in range(len(spec.PARTICIPATION_FLAG_WEIGHTS)):
        full_flags = spec.add_flag(full_flags, flag_index)
    return full_flags

def run_process_participation_flag_updates(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    old = state.current_epoch_participation.copy()
    yield from run_epoch_processing_with(spec, state, 'process_participation_flag_updates')
    assert state.current_epoch_participation == [0] * len(state.validators)
    assert state.previous_epoch_participation == old

@with_altair_and_later
@spec_state_test
def test_all_zeroed(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    state.current_epoch_participation = [0] * len(state.validators)
    state.previous_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_filled(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    state.previous_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    state.current_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_previous_filled(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    state.previous_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    state.current_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_current_filled(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    state.previous_epoch_participation = [0] * len(state.validators)
    state.current_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

def random_flags(spec: Any, state: Any, seed: int, previous: bool = True, current: bool = True) -> None:
    rng = Random(seed)
    count = len(state.validators)
    max_flag_value_excl = 2 ** len(spec.PARTICIPATION_FLAG_WEIGHTS)
    if previous:
        state.previous_epoch_participation = [rng.randrange(0, max_flag_value_excl) for _ in range(count)]
    if current:
        state.current_epoch_participation = [rng.randrange(0, max_flag_value_excl) for _ in range(count)]

@with_altair_and_later
@spec_state_test
def test_random_0(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 100)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_random_1(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 101)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_random_2(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 102)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_random_genesis(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    random_flags(spec, state, 11)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_current_epoch_zeroed(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 12, current=False)
    state.current_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_previous_epoch_zeroed(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 13, previous=False)
    state.previous_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

def custom_validator_count(factor: float) -> Callable[[Any], List[int]]:
    def initializer(spec: Any) -> List[int]:
        num_validators = spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
        return [spec.MAX_EFFECTIVE_BALANCE] * int(float(int(num_validators)) * factor)
    return initializer

@with_altair_and_later
@with_presets([MINIMAL], reason='mainnet config requires too many pre-generated public/private keys')
@spec_test
@with_custom_state(balances_fn=custom_validator_count(1.3), threshold_fn=lambda spec: spec.config.EJECTION_BALANCE)
@single_phase
def test_slightly_larger_random(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 14)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@with_presets([MINIMAL], reason='mainnet config requires too many pre-generated public/private keys')
@spec_test
@with_custom_state(balances_fn=custom_validator_count(2.6), threshold_fn=lambda spec: spec.config.EJECTION_BALANCE)
@single_phase
def test_large_random(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 15)
    yield from run_process_participation_flag_updates(spec, state)
