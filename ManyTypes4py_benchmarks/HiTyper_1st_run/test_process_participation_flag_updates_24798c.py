from random import Random
from eth2spec.test.helpers.constants import MINIMAL
from eth2spec.test.context import with_altair_and_later, with_custom_state, spec_test, spec_state_test, with_presets, single_phase
from eth2spec.test.helpers.state import next_epoch_via_block
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with

def get_full_flags(spec: Union[int, dict[int, int]]) -> int:
    full_flags = spec.ParticipationFlags(0)
    for flag_index in range(len(spec.PARTICIPATION_FLAG_WEIGHTS)):
        full_flags = spec.add_flag(full_flags, flag_index)
    return full_flags

def run_process_participation_flag_updates(spec: Union[dict, prefecengine.state.State, dict[typing.Type, typing.Any]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, State]) -> typing.Generator:
    old = state.current_epoch_participation.copy()
    yield from run_epoch_processing_with(spec, state, 'process_participation_flag_updates')
    assert state.current_epoch_participation == [0] * len(state.validators)
    assert state.previous_epoch_participation == old

@with_altair_and_later
@spec_state_test
def test_all_zeroed(spec: Union[dict["core.Edge", "state.State"], prefecengine.state.State, dict[str, typing.Any]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, mythril.laser.ethereum.state.constraints.Constraints, prefecengine.state.State]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    state.current_epoch_participation = [0] * len(state.validators)
    state.previous_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_filled(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"], prefecengine.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"], prefecengine.state.State]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    state.previous_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    state.current_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_previous_filled(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[dict["core.Edge", "state.State"], mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    state.previous_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    state.current_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_current_filled(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, list[mythril.laser.ethereum.state.global_state.GlobalState]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    state.previous_epoch_participation = [0] * len(state.validators)
    state.current_epoch_participation = [get_full_flags(spec)] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

def random_flags(spec: Union[int, list[int], typing.Sequence], state: Union[int, list[int]], seed: Union[int, random.Random, mythril.laser.ethereum.state.global_state.GlobalState], previous: bool=True, current: bool=True) -> None:
    rng = Random(seed)
    count = len(state.validators)
    max_flag_value_excl = 2 ** len(spec.PARTICIPATION_FLAG_WEIGHTS)
    if previous:
        state.previous_epoch_participation = [rng.randrange(0, max_flag_value_excl) for _ in range(count)]
    if current:
        state.current_epoch_participation = [rng.randrange(0, max_flag_value_excl) for _ in range(count)]

@with_altair_and_later
@spec_state_test
def test_random_0(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 100)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_random_1(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 101)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_random_2(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 102)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_random_genesis(spec: Union[int, dict, str], state: Union[int, dict, str]) -> typing.Generator:
    random_flags(spec, state, 11)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_current_epoch_zeroed(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, aiocometd.constants.TransportState, dict["core.Edge", "state.State"]], state: Union[prefecengine.state.State, mythril.laser.ethereum.state.global_state.GlobalState, aiocometd.constants.TransportState]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 12, current=False)
    state.current_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_previous_epoch_zeroed(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, aiocometd.constants.TransportState], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 13, previous=False)
    state.previous_epoch_participation = [0] * len(state.validators)
    yield from run_process_participation_flag_updates(spec, state)

def custom_validator_count(factor: Union[float, int, T]):

    def initializer(spec: Any) -> list:
        num_validators = spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
        return [spec.MAX_EFFECTIVE_BALANCE] * int(float(int(num_validators)) * factor)
    return initializer

@with_altair_and_later
@with_presets([MINIMAL], reason='mainnet config requires too many pre-generated public/private keys')
@spec_test
@with_custom_state(balances_fn=custom_validator_count(1.3), threshold_fn=lambda spec: spec.config.EJECTION_BALANCE)
@single_phase
def test_slightly_larger_random(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, raiden.transfer.state.ChainState], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, raiden.transfer.state.ChainState]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 14)
    yield from run_process_participation_flag_updates(spec, state)

@with_altair_and_later
@with_presets([MINIMAL], reason='mainnet config requires too many pre-generated public/private keys')
@spec_test
@with_custom_state(balances_fn=custom_validator_count(2.6), threshold_fn=lambda spec: spec.config.EJECTION_BALANCE)
@single_phase
def test_large_random(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]]) -> typing.Generator:
    next_epoch_via_block(spec, state)
    random_flags(spec, state, 15)
    yield from run_process_participation_flag_updates(spec, state)