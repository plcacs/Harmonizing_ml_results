from random import Random
from typing import Optional, List
from eth2spec.test.helpers.attestations import cached_prepare_state_with_attestations
from eth2spec.test.helpers.deposits import mock_deposit
from eth2spec.test.helpers.forks import is_post_altair
from eth2spec.test.helpers.state import next_epoch

def set_some_activations(
    spec: Spec,
    state: State,
    rng: Random,
    activation_epoch: Optional[Epoch] = None
) -> List[int]:
    if activation_epoch is None:
        activation_epoch = spec.get_current_epoch(state)
    num_validators = len(state.validators)
    selected_indices: List[int] = []
    for index in range(num_validators):
        if state.validators[index].slashed or state.validators[index].exit_epoch != spec.FAR_FUTURE_EPOCH:
            continue
        if rng.randrange(num_validators) < num_validators // 10:
            state.validators[index].activation_eligibility_epoch = max(
                int(activation_epoch) - int(spec.MAX_SEED_LOOKAHEAD) - 1,
                spec.GENESIS_EPOCH
            )
            state.validators[index].activation_epoch = activation_epoch
            selected_indices.append(index)
    return selected_indices

def set_some_new_deposits(
    spec: Spec,
    state: State,
    rng: Random
) -> List[int]:
    deposited_indices: List[int] = []
    num_validators = len(state.validators)
    for index in range(num_validators):
        if not spec.is_active_validator(state.validators[index], spec.get_current_epoch(state)):
            continue
        if rng.randrange(num_validators) < num_validators // 10:
            mock_deposit(spec, state, index)
            if rng.choice([True, False]):
                state.validators[index].activation_eligibility_epoch = spec.get_current_epoch(state)
            else:
                deposited_indices.append(index)
    return deposited_indices

def exit_random_validators(
    spec: Spec,
    state: State,
    rng: Random,
    fraction: float = 0.5,
    exit_epoch: Optional[Epoch] = None,
    withdrawable_epoch: Optional[Epoch] = None,
    from_epoch: Optional[Epoch] = None
) -> List[int]:
    """
    Set some validators' exit_epoch and withdrawable_epoch.

    If exit_epoch is configured, use the given exit_epoch. Otherwise, randomly set exit_epoch and withdrawable_epoch.
    """
    if from_epoch is None:
        from_epoch = spec.MAX_SEED_LOOKAHEAD + 1
    epoch_diff = int(from_epoch) - int(spec.get_current_epoch(state))
    for _ in range(epoch_diff):
        next_epoch(spec, state)
    current_epoch = spec.get_current_epoch(state)
    exited_indices: List[int] = []
    for index in spec.get_active_validator_indices(state, current_epoch):
        sampled = rng.random() < fraction
        if not sampled:
            continue
        exited_indices.append(index)
        validator = state.validators[index]
        if exit_epoch is None:
            assert withdrawable_epoch is None
            validator.exit_epoch = rng.choice([
                current_epoch,
                current_epoch - 1,
                current_epoch - 2,
                current_epoch - 3
            ])
            if rng.choice([True, False]):
                validator.withdrawable_epoch = current_epoch
            else:
                validator.withdrawable_epoch = current_epoch + 1
        else:
            validator.exit_epoch = exit_epoch
            if withdrawable_epoch is None:
                validator.withdrawable_epoch = validator.exit_epoch + spec.config.MIN_VALIDATOR_WITHDRAWABILITY_DELAY
            else:
                validator.withdrawable_epoch = withdrawable_epoch
    return exited_indices

def slash_random_validators(
    spec: Spec,
    state: State,
    rng: Random,
    fraction: float = 0.5
) -> List[int]:
    slashed_indices: List[int] = []
    for index in range(len(state.validators)):
        sampled = rng.random() < fraction
        if index == 0 or sampled:
            spec.slash_validator(state, index)
            slashed_indices.append(index)
    return slashed_indices

def randomize_epoch_participation(
    spec: Spec,
    state: State,
    epoch: Epoch,
    rng: Random
) -> None:
    assert epoch in (spec.get_current_epoch(state), spec.get_previous_epoch(state))
    if not is_post_altair(spec):
        if epoch == spec.get_current_epoch(state):
            pending_attestations = state.current_epoch_attestations
        else:
            pending_attestations = state.previous_epoch_attestations
        for pending_attestation in pending_attestations:
            if rng.randint(0, 2) == 0:
                pending_attestation.data.target.root = b'U' * 32
            if rng.randint(0, 2) == 0:
                pending_attestation.data.beacon_block_root = b'f' * 32
            pending_attestation.aggregation_bits = [
                rng.choice([True, False]) for _ in pending_attestation.aggregation_bits
            ]
            pending_attestation.inclusion_delay = rng.randint(1, spec.SLOTS_PER_EPOCH)
    else:
        if epoch == spec.get_current_epoch(state):
            epoch_participation = state.current_epoch_participation
        else:
            epoch_participation = state.previous_epoch_participation
        for index in range(len(state.validators)):
            is_timely_correct_head = rng.randint(0, 2) != 0
            flags = epoch_participation[index]

            def set_flag(flag_index: int, value: bool) -> None:
                nonlocal flags
                flag = spec.ParticipationFlags(2 ** flag_index)
                if value:
                    flags |= flag
                else:
                    flags &= 255 ^ flag
            set_flag(spec.TIMELY_HEAD_FLAG_INDEX, is_timely_correct_head)
            if is_timely_correct_head:
                set_flag(spec.TIMELY_TARGET_FLAG_INDEX, True)
                set_flag(spec.TIMELY_SOURCE_FLAG_INDEX, True)
            else:
                set_flag(spec.TIMELY_TARGET_FLAG_INDEX, rng.choice([True, False]))
                set_flag(spec.TIMELY_SOURCE_FLAG_INDEX, rng.choice([True, False]))
            epoch_participation[index] = flags

def randomize_previous_epoch_participation(
    spec: Spec,
    state: State,
    rng: Random = Random(8020)
) -> None:
    cached_prepare_state_with_attestations(spec, state)
    randomize_epoch_participation(spec, state, spec.get_previous_epoch(state), rng)
    if not is_post_altair(spec):
        state.current_epoch_attestations = []
    else:
        state.current_epoch_participation = [
            spec.ParticipationFlags(0) for _ in range(len(state.validators))
        ]

def randomize_attestation_participation(
    spec: Spec,
    state: State,
    rng: Random = Random(8020)
) -> None:
    cached_prepare_state_with_attestations(spec, state)
    randomize_epoch_participation(spec, state, spec.get_previous_epoch(state), rng)
    randomize_epoch_participation(spec, state, spec.get_current_epoch(state), rng)

def randomize_state(
    spec: Spec,
    state: State,
    rng: Random = Random(8020),
    exit_fraction: float = 0.5,
    slash_fraction: float = 0.5
) -> None:
    set_some_new_deposits(spec, state, rng)
    exit_random_validators(spec, state, rng, fraction=exit_fraction)
    slash_random_validators(spec, state, rng, fraction=slash_fraction)
    randomize_attestation_participation(spec, state, rng)

def patch_state_to_non_leaking(
    spec: Spec,
    state: State
) -> None:
    """
    This function performs an irregular state transition so that:
    1. the current justified checkpoint references the previous epoch
    2. the previous justified checkpoint references the epoch before previous
    3. the finalized checkpoint matches the previous justified checkpoint

    The effects of this function are intended to offset randomization side effects
    performed by other functionality in this module so that if the ``state`` was leaking,
    then the ``state`` is not leaking after.
    """
    state.justification_bits[0] = True
    state.justification_bits[1] = True
    previous_epoch = spec.get_previous_epoch(state)
    previous_root = spec.get_block_root(state, previous_epoch)
    previous_previous_epoch = max(spec.GENESIS_EPOCH, spec.Epoch(previous_epoch - 1))
    previous_previous_root = spec.get_block_root(state, previous_previous_epoch)
    state.previous_justified_checkpoint = spec.Checkpoint(epoch=previous_previous_epoch, root=previous_previous_root)
    state.current_justified_checkpoint = spec.Checkpoint(epoch=previous_epoch, root=previous_root)
    state.finalized_checkpoint = spec.Checkpoint(epoch=previous_previous_epoch, root=previous_previous_root)
