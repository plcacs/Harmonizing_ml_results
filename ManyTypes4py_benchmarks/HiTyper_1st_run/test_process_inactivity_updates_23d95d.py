from random import Random
from eth2spec.test.context import spec_state_test, with_altair_and_later
from eth2spec.test.helpers.inactivity_scores import randomize_inactivity_scores, zero_inactivity_scores
from eth2spec.test.helpers.state import next_epoch, next_epoch_via_block, set_full_participation, set_empty_participation
from eth2spec.test.helpers.voluntary_exits import exit_validators, get_exited_validators
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with
from eth2spec.test.helpers.random import randomize_attestation_participation, randomize_previous_epoch_participation, randomize_state
from eth2spec.test.helpers.rewards import leaking

def run_process_inactivity_updates(spec: Union[dict, prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[dict, prefecengine.state.State, dict["core.Edge", "state.State"]]) -> typing.Generator:
    yield from run_epoch_processing_with(spec, state, 'process_inactivity_updates')

@with_altair_and_later
@spec_state_test
def test_genesis(spec: Union[dict, mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"]], state: Union[dict, mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"]]) -> typing.Generator:
    yield from run_process_inactivity_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_genesis_random_scores(spec: Union[prefecengine.state.State, prefecengine.state.State, list[mythril.laser.ethereum.state.global_state.GlobalState]], state: Union[dict, state.State]) -> typing.Generator:
    rng = Random(10102)
    state.inactivity_scores = [rng.randint(0, 100) for _ in state.inactivity_scores]
    pre_scores = state.inactivity_scores.copy()
    yield from run_process_inactivity_updates(spec, state)
    assert state.inactivity_scores == pre_scores

def run_inactivity_scores_test(spec: Any, state: Any, participation_fn: Union[None, list[float], float]=None, inactivity_scores_fn: Union[None, float, typing.Callable[str, float], dict[str, typing.Any]]=None, rng: Random=Random(10101) -> typing.Generator):
    while True:
        try:
            next_epoch_via_block(spec, state)
        except AssertionError:
            next_epoch(spec, state)
        else:
            break
    if participation_fn is not None:
        participation_fn(spec, state, rng=rng)
    if inactivity_scores_fn is not None:
        inactivity_scores_fn(spec, state, rng=rng)
    yield from run_process_inactivity_updates(spec, state)

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_empty_participation(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict[str, float], int], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict[str, typing.Any]]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_empty_participation, zero_inactivity_scores)
    assert set(state.inactivity_scores) == set([0])

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_empty_participation_leaking(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict[str, float]], state: Union[dict[str, float], mythril.laser.ethereum.state.global_state.GlobalState]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_empty_participation, zero_inactivity_scores)
    assert spec.is_in_inactivity_leak(state)
    for score in state.inactivity_scores:
        assert score > 0

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_random_participation(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, apps.monero.signing.state.State, bytes], state: Union[dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState, pyshgp.push.state.PushState]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, randomize_attestation_participation, zero_inactivity_scores, rng=Random(5555))
    assert set(state.inactivity_scores) == set([0])

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_random_participation_leaking(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, apps.monero.signing.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, apps.monero.signing.state.State]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, randomize_previous_epoch_participation, zero_inactivity_scores, rng=Random(5555))
    assert spec.is_in_inactivity_leak(state)
    assert 0 in state.inactivity_scores
    assert len(set(state.inactivity_scores)) > 1

@with_altair_and_later
@spec_state_test
def test_all_zero_inactivity_scores_full_participation(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict[str, str]], state: Union[dict[str, typing.Any], prefecengine.state.State, dict]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_full_participation, zero_inactivity_scores)
    assert set(state.inactivity_scores) == set([0])

@with_altair_and_later
@spec_state_test
@leaking()
def test_all_zero_inactivity_scores_full_participation_leaking(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict[str, float]], state: Union[prefecengine.state.State, mythril.laser.ethereum.state.global_state.GlobalState, dict[str, typing.Any]]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_full_participation, zero_inactivity_scores)
    assert spec.is_in_inactivity_leak(state)
    assert set(state.inactivity_scores) == set([0])

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_empty_participation(spec: Union[bytes, apps.monero.signing.state.State, dict[str, float]], state: Union[bytes, apps.monero.signing.state.State, dict[str, float]]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_empty_participation, randomize_inactivity_scores, Random(9999))

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_empty_participation_leaking(spec: Union[apps.monero.signing.state.State, int], state: Union[apps.monero.signing.state.State, int]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_empty_participation, randomize_inactivity_scores, Random(9999))
    assert spec.is_in_inactivity_leak(state)

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_random_participation(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, bytes, apps.monero.signing.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, bytes, apps.monero.signing.state.State]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, randomize_attestation_participation, randomize_inactivity_scores, Random(22222))

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_random_participation_leaking(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, apps.monero.signing.state.State, bytes], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, apps.monero.signing.state.State, bytes]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, randomize_previous_epoch_participation, randomize_inactivity_scores, Random(22222))
    assert spec.is_in_inactivity_leak(state)

@with_altair_and_later
@spec_state_test
def test_random_inactivity_scores_full_participation(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict[str, float], int], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict[str, float], int]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_full_participation, randomize_inactivity_scores, Random(33333))

@with_altair_and_later
@spec_state_test
@leaking()
def test_random_inactivity_scores_full_participation_leaking(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, pyshgp.push.state.PushState, dict[str, float]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, pyshgp.push.state.PushState, dict[str, float]]) -> typing.Generator:
    yield from run_inactivity_scores_test(spec, state, set_full_participation, randomize_inactivity_scores, Random(33333))
    assert spec.is_in_inactivity_leak(state)

def slash_some_validators_for_inactivity_scores_test(spec: Union[prefecengine.state.State, mythril.laser.ethereum.state.constraints.Constraints], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, list[float]], rng: Random=Random(40404040) -> None):
    future_state = state.copy()
    next_epoch_via_block(spec, future_state)
    proposer_index = spec.get_beacon_proposer_index(future_state)
    for validator_index in range(len(state.validators)):
        if rng.choice(range(4)) == 0 and validator_index != proposer_index:
            spec.slash_validator(state, validator_index)

@with_altair_and_later
@spec_state_test
def test_some_slashed_zero_scores_full_participation(spec: mythril.laser.ethereum.state.global_state.GlobalState, state: Union[pyshgp.push.state.PushState, dict, mythril.laser.ethereum.state.global_state.GlobalState]) -> typing.Generator:
    slash_some_validators_for_inactivity_scores_test(spec, state, rng=Random(33429))
    yield from run_inactivity_scores_test(spec, state, set_full_participation, zero_inactivity_scores)
    assert set(state.inactivity_scores) == set([0])

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_slashed_zero_scores_full_participation_leaking(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"]]) -> typing.Generator:
    slash_some_validators_for_inactivity_scores_test(spec, state, rng=Random(332243))
    yield from run_inactivity_scores_test(spec, state, set_full_participation, zero_inactivity_scores)
    assert spec.is_in_inactivity_leak(state)
    for score, validator in zip(state.inactivity_scores, state.validators):
        if validator.slashed:
            assert score > 0
        else:
            assert score == 0

@with_altair_and_later
@spec_state_test
def test_some_slashed_full_random(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, pyshgp.push.state.PushState, int], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, pyshgp.push.state.PushState, int]) -> typing.Generator:
    rng = Random(1010222)
    slash_some_validators_for_inactivity_scores_test(spec, state, rng=rng)
    yield from run_inactivity_scores_test(spec, state, randomize_attestation_participation, randomize_inactivity_scores, rng=rng)

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_slashed_full_random_leaking(spec: mythril.laser.ethereum.state.global_state.GlobalState, state: mythril.laser.ethereum.state.global_state.GlobalState) -> typing.Generator:
    rng = Random(1102233)
    slash_some_validators_for_inactivity_scores_test(spec, state, rng=rng)
    yield from run_inactivity_scores_test(spec, state, randomize_previous_epoch_participation, randomize_inactivity_scores, rng=rng)
    assert spec.is_in_inactivity_leak(state)

@with_altair_and_later
@spec_state_test
@leaking()
def test_some_exited_full_random_leaking(spec: Union[dict[str, typing.Any], str, list], state: Union[dict[str, typing.Any], str, list]) -> typing.Generator:
    rng = Random(1102233)
    exit_count = 3
    randomize_inactivity_scores(spec, state, rng=rng)
    assert not any(get_exited_validators(spec, state))
    exited_indices = exit_validators(spec, state, exit_count, rng=rng)
    assert not any(get_exited_validators(spec, state))
    target_epoch = max((state.validators[index].exit_epoch for index in exited_indices))
    previous_epoch = spec.get_previous_epoch(state)
    for _ in range(target_epoch - previous_epoch):
        next_epoch(spec, state)
    assert len(get_exited_validators(spec, state)) == exit_count
    previous_scores = state.inactivity_scores.copy()
    yield from run_inactivity_scores_test(spec, state, randomize_previous_epoch_participation, rng=rng)
    some_changed = False
    for index in range(len(state.validators)):
        if index in exited_indices:
            assert previous_scores[index] == state.inactivity_scores[index]
        else:
            previous_score = previous_scores[index]
            current_score = state.inactivity_scores[index]
            if previous_score != current_score:
                some_changed = True
    assert some_changed
    assert spec.is_in_inactivity_leak(state)

def _run_randomized_state_test_for_inactivity_updates(spec: Union[dict[str, float], state.state.State], state: Any, rng: Random=Random(13377331) -> typing.Generator):
    randomize_inactivity_scores(spec, state, rng=rng)
    randomize_state(spec, state, rng=rng)
    exited_validators = get_exited_validators(spec, state)
    exited_but_not_slashed = []
    for index in exited_validators:
        validator = state.validators[index]
        if validator.slashed:
            continue
        exited_but_not_slashed.append(index)
    assert len(exited_but_not_slashed) > 0
    some_exited_validator = exited_but_not_slashed[0]
    pre_score_for_exited_validator = state.inactivity_scores[some_exited_validator]
    assert pre_score_for_exited_validator != 0
    assert len(set(state.inactivity_scores)) > 1
    yield from run_inactivity_scores_test(spec, state)
    post_score_for_exited_validator = state.inactivity_scores[some_exited_validator]
    assert pre_score_for_exited_validator == post_score_for_exited_validator

@with_altair_and_later
@spec_state_test
def test_randomized_state(spec: dict[str, float], state: dict[str, float]) -> typing.Generator:
    """
    This test ensures that once a validator has exited,
    their inactivity score does not change.
    """
    rng = Random(10011001)
    yield from _run_randomized_state_test_for_inactivity_updates(spec, state, rng=rng)

@with_altair_and_later
@spec_state_test
@leaking()
def test_randomized_state_leaking(spec: Union[pyshgp.push.state.PushState, dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState], state: Union[pyshgp.push.state.PushState, dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState]) -> typing.Generator:
    """
    This test ensures that once a validator has exited,
    their inactivity score does not change, even during a leak.
    Note that slashed validators are still subject to mutations
    (refer ``get_eligible_validator_indices`).
    """
    rng = Random(10011002)
    yield from _run_randomized_state_test_for_inactivity_updates(spec, state, rng=rng)
    assert spec.is_in_inactivity_leak(state)