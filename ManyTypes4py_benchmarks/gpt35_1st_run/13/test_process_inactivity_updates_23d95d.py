from random import Random
from eth2spec.test.context import spec_state_test, with_altair_and_later
from eth2spec.test.helpers.inactivity_scores import randomize_inactivity_scores, zero_inactivity_scores
from eth2spec.test.helpers.state import next_epoch, next_epoch_via_block, set_full_participation, set_empty_participation
from eth2spec.test.helpers.voluntary_exits import exit_validators, get_exited_validators
from eth2spec.test.helpers.epoch_processing import run_epoch_processing_with
from eth2spec.test.helpers.random import randomize_attestation_participation, randomize_previous_epoch_participation, randomize_state
from eth2spec.test.helpers.rewards import leaking
from typing import Iterator

def run_process_inactivity_updates(spec, state) -> Iterator:
    yield from run_epoch_processing_with(spec, state, 'process_inactivity_updates')

def run_inactivity_scores_test(spec, state, participation_fn=None, inactivity_scores_fn=None, rng=Random(10101)) -> Iterator:
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

def slash_some_validators_for_inactivity_scores_test(spec, state, rng=Random(40404040)) -> None:
    future_state = state.copy()
    next_epoch_via_block(spec, future_state)
    proposer_index = spec.get_beacon_proposer_index(future_state)
    for validator_index in range(len(state.validators)):
        if rng.choice(range(4)) == 0 and validator_index != proposer_index:
            spec.slash_validator(state, validator_index)

def _run_randomized_state_test_for_inactivity_updates(spec, state, rng=Random(13377331)) -> Iterator:
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

