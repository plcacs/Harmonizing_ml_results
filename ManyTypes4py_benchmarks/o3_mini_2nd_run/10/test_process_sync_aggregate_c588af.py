#!/usr/bin/env python3
import random
from typing import Iterator, Any, List, Callable

from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.state import (
    state_transition_and_sign_block,
    transition_to,
    next_epoch_via_block,
)
from eth2spec.test.helpers.constants import MAINNET, MINIMAL
from eth2spec.test.helpers.sync_committee import (
    compute_aggregate_sync_committee_signature,
    compute_committee_indices,
    run_sync_committee_processing,
    run_successful_sync_committee_test,
)
from eth2spec.test.helpers.voluntary_exits import get_unslashed_exited_validators
from eth2spec.test.context import with_altair_and_later, with_presets, spec_state_test, always_bls, single_phase, with_custom_state, spec_test, default_balances_electra, default_activation_threshold


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_bad_domain(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, committee_indices, block_root=block.parent_root, domain_type=spec.DOMAIN_BEACON_ATTESTER
        ),
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_missing_participant(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    rng: random.Random = random.Random(2020)
    random_participant: int = rng.choice(committee_indices)
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[index != random_participant for index in committee_indices],
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, committee_indices, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_no_participants(spec: Any, state: Any) -> Iterator[Any]:
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[False] * len(block.body.sync_aggregate.sync_committee_bits),
        sync_committee_signature=b'\x00' * 96,
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_infinite_signature_with_all_participants(spec: Any, state: Any) -> Iterator[Any]:
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(block.body.sync_committee.sync_committee_bits),
        sync_committee_signature=spec.G2_POINT_AT_INFINITY,
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_infinite_signature_with_single_participant(spec: Any, state: Any) -> Iterator[Any]:
    block = build_empty_block_for_next_slot(spec, state)
    bits: List[bool] = [True] + [False] * (len(block.body.sync_committee.sync_committee_bits) - 1)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=bits,
        sync_committee_signature=spec.G2_POINT_AT_INFINITY,
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_extra_participant(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    rng: random.Random = random.Random(3030)
    random_participant: int = rng.choice(committee_indices)
    block = build_empty_block_for_next_slot(spec, state)
    extra_committee: List[int] = [index for index in committee_indices if index != random_participant]
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, extra_committee, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


def is_duplicate_sync_committee(committee_indices: List[int]) -> bool:
    dup = {v for v in committee_indices if committee_indices.count(v) > 1}
    return len(dup) > 0


@with_altair_and_later
@with_presets([MINIMAL], reason='to create nonduplicate committee')
@spec_test
@with_custom_state(balances_fn=default_balances_electra, threshold_fn=default_activation_threshold)
@single_phase
def test_sync_committee_rewards_nonduplicate_committee(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    assert not is_duplicate_sync_committee(committee_indices)
    committee_size: int = len(committee_indices)
    committee_bits: List[bool] = [True] * committee_size
    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@with_presets([MAINNET], reason='to create duplicate committee')
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_no_participation(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    committee_size: int = len(committee_indices)
    committee_bits: List[bool] = [False] * committee_size
    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@with_presets([MAINNET], reason='to create duplicate committee')
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_half_participation(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    committee_size: int = len(committee_indices)
    committee_bits: List[bool] = [True] * (committee_size // 2) + [False] * (committee_size // 2)
    assert len(committee_bits) == committee_size
    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@with_presets([MAINNET], reason='to create duplicate committee')
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_full_participation(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    committee_size: int = len(committee_indices)
    committee_bits: List[bool] = [True] * committee_size
    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


def _run_sync_committee_selected_twice(
    spec: Any,
    state: Any,
    pre_balance: int,
    participate_first_position: bool,
    participate_second_position: bool,
    skip_reward_validation: bool = False,
) -> Iterator[int]:
    committee_indices: List[int] = compute_committee_indices(state)
    assert is_duplicate_sync_committee(committee_indices)
    committee_size: int = len(committee_indices)
    committee_bits: List[bool] = [False] * committee_size
    dup: set = {v for v in committee_indices if committee_indices.count(v) == 2}
    assert len(dup) > 0
    validator_index: int = dup.pop()
    positions: List[int] = [i for i, v in enumerate(committee_indices) if v == validator_index]
    committee_bits[positions[0]] = participate_first_position
    committee_bits[positions[1]] = participate_second_position
    state.balances[validator_index] = pre_balance
    state.validators[validator_index].effective_balance = min(
        pre_balance - pre_balance % spec.EFFECTIVE_BALANCE_INCREMENT, spec.MAX_EFFECTIVE_BALANCE
    )
    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits, skip_reward_validation=skip_reward_validation)
    return validator_index


@with_altair_and_later
@with_presets([MAINNET], reason='to create duplicate committee')
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_zero_balance_only_participate_first_one(
    spec: Any, state: Any
) -> Iterator[Any]:
    validator_index: int = (yield from _run_sync_committee_selected_twice(
        spec, state, pre_balance=0, participate_first_position=True, participate_second_position=False
    ))
    assert state.balances[validator_index] == 0
    yield  # To satisfy generator return type


@with_altair_and_later
@with_presets([MAINNET], reason='to create duplicate committee')
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_zero_balance_only_participate_second_one(
    spec: Any, state: Any
) -> Iterator[Any]:
    validator_index: int = (yield from _run_sync_committee_selected_twice(
        spec, state, pre_balance=0, participate_first_position=False, participate_second_position=True, skip_reward_validation=True
    ))
    assert state.balances[validator_index] > 0
    yield


@with_altair_and_later
@with_presets([MAINNET], reason='to create duplicate committee')
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_max_effective_balance_only_participate_first_one(
    spec: Any, state: Any
) -> Iterator[Any]:
    validator_index: int = (yield from _run_sync_committee_selected_twice(
        spec, state, pre_balance=spec.MAX_EFFECTIVE_BALANCE, participate_first_position=True, participate_second_position=False
    ))
    assert state.balances[validator_index] == spec.MAX_EFFECTIVE_BALANCE
    yield


@with_altair_and_later
@with_presets([MAINNET], reason='to create duplicate committee')
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_max_effective_balance_only_participate_second_one(
    spec: Any, state: Any
) -> Iterator[Any]:
    validator_index: int = (yield from _run_sync_committee_selected_twice(
        spec, state, pre_balance=spec.MAX_EFFECTIVE_BALANCE, participate_first_position=False, participate_second_position=True
    ))
    assert state.balances[validator_index] == spec.MAX_EFFECTIVE_BALANCE
    yield


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_rewards_not_full_participants(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    rng: random.Random = random.Random(1010)
    committee_bits: List[bool] = [rng.choice([True, False]) for _ in committee_indices]
    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_rewards_empty_participants(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    committee_bits: List[bool] = [False for _ in committee_indices]
    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_past_block(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state)
    for _ in range(2):
        block = build_empty_block_for_next_slot(spec, state)
        block.body.sync_aggregate = spec.SyncAggregate(
            sync_committee_bits=[True] * len(committee_indices),
            sync_committee_signature=compute_aggregate_sync_committee_signature(
                spec, state, block.slot - 1, committee_indices, block_root=block.parent_root
            ),
        )
        state_transition_and_sign_block(spec, state, block)
    invalid_block = build_empty_block_for_next_slot(spec, state)
    invalid_block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, invalid_block.slot - 2, committee_indices
        ),
    )
    yield from run_sync_committee_processing(spec, state, invalid_block, expect_exception=True)


@with_altair_and_later
@with_presets([MINIMAL], reason='to produce different committee sets')
@spec_state_test
@always_bls
def test_invalid_signature_previous_committee(spec: Any, state: Any) -> Iterator[Any]:
    current_epoch: int = spec.get_current_epoch(state)
    old_sync_committee: Any = state.next_sync_committee
    epoch_in_future_sync_committee_period: int = current_epoch + 2 * spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD
    slot_in_future_sync_committee_period: int = epoch_in_future_sync_committee_period * spec.SLOTS_PER_EPOCH
    transition_to(spec, state, slot_in_future_sync_committee_period)
    assert set(old_sync_committee.pubkeys) != set(state.current_sync_committee.pubkeys)
    committee_indices: List[int] = compute_committee_indices(state, old_sync_committee)
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, committee_indices, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
@with_presets([MINIMAL], reason='too slow')
def test_valid_signature_future_committee(spec: Any, state: Any) -> Iterator[Any]:
    current_epoch: int = spec.get_current_epoch(state)
    old_current_sync_committee: Any = state.current_sync_committee
    old_next_sync_committee: Any = state.next_sync_committee
    epoch_in_future_sync_committee_period: int = current_epoch + 2 * spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD
    slot_in_future_sync_committee_period: int = epoch_in_future_sync_committee_period * spec.SLOTS_PER_EPOCH
    transition_to(spec, state, slot_in_future_sync_committee_period)
    sync_committee: Any = state.current_sync_committee
    next_sync_committee: Any = state.next_sync_committee
    assert next_sync_committee != sync_committee
    assert sync_committee != old_current_sync_committee
    assert sync_committee != old_next_sync_committee
    committee_indices: List[int] = compute_committee_indices(state, sync_committee)
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, committee_indices, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block)


@with_altair_and_later
@spec_state_test
@always_bls
@with_presets([MINIMAL], reason='prefer short search to find matching proposer')
def test_proposer_in_committee_without_participation(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state, state.current_sync_committee)
    for _ in range(spec.SLOTS_PER_EPOCH):
        block = build_empty_block_for_next_slot(spec, state)
        proposer_index: int = block.proposer_index
        proposer_pubkey: Any = state.validators[proposer_index].pubkey
        proposer_is_in_sync_committee: bool = proposer_pubkey in state.current_sync_committee.pubkeys
        if proposer_is_in_sync_committee:
            participation: List[bool] = [index != proposer_index for index in committee_indices]
            participants: List[int] = [index for index in committee_indices if index != proposer_index]
        else:
            participation = [True for _ in committee_indices]
            participants = committee_indices
        block.body.sync_aggregate = spec.SyncAggregate(
            sync_committee_bits=participation,
            sync_committee_signature=compute_aggregate_sync_committee_signature(
                spec, state, block.slot - 1, participants, block_root=block.parent_root
            ),
        )
        if proposer_is_in_sync_committee:
            assert state.validators[block.proposer_index].pubkey in state.current_sync_committee.pubkeys
            yield from run_sync_committee_processing(spec, state, block)
            break
        else:
            state_transition_and_sign_block(spec, state, block)
    else:
        raise AssertionError('failed to find a proposer in the sync committee set; check test setup')
    yield


@with_altair_and_later
@spec_state_test
@always_bls
@with_presets([MINIMAL], reason='prefer short search to find matching proposer')
def test_proposer_in_committee_with_participation(spec: Any, state: Any) -> Iterator[Any]:
    committee_indices: List[int] = compute_committee_indices(state, state.current_sync_committee)
    participation: List[bool] = [True for _ in committee_indices]
    for _ in range(spec.SLOTS_PER_EPOCH):
        block = build_empty_block_for_next_slot(spec, state)
        proposer_index: int = block.proposer_index
        proposer_pubkey: Any = state.validators[proposer_index].pubkey
        proposer_is_in_sync_committee: bool = proposer_pubkey in state.current_sync_committee.pubkeys
        block.body.sync_aggregate = spec.SyncAggregate(
            sync_committee_bits=participation,
            sync_committee_signature=compute_aggregate_sync_committee_signature(
                spec, state, block.slot - 1, committee_indices, block_root=block.parent_root
            ),
        )
        if proposer_is_in_sync_committee:
            assert state.validators[block.proposer_index].pubkey in state.current_sync_committee.pubkeys
            yield from run_sync_committee_processing(spec, state, block)
            return
        else:
            state_transition_and_sign_block(spec, state, block)
    raise AssertionError('failed to find a proposer in the sync committee set; check test setup')


def _exit_validator_from_committee_and_transition_state(
    spec: Any,
    state: Any,
    committee_indices: List[int],
    rng: random.Random,
    target_epoch_provider: Callable[[Any], int],
    withdrawable_offset: int = 1,
) -> int:
    exited_validator_index: int = rng.sample(committee_indices, 1)[0]
    validator: Any = state.validators[exited_validator_index]
    current_epoch: int = spec.get_current_epoch(state)
    validator.exit_epoch = current_epoch
    validator.withdrawable_epoch = validator.exit_epoch + withdrawable_offset
    target_epoch: int = target_epoch_provider(state.validators[exited_validator_index])
    target_slot: int = target_epoch * spec.SLOTS_PER_EPOCH
    transition_to(spec, state, target_slot)
    exited_validator_indices: List[int] = get_unslashed_exited_validators(spec, state)
    assert exited_validator_index in exited_validator_indices
    exited_pubkey: Any = state.validators[exited_validator_index].pubkey
    assert exited_pubkey in state.current_sync_committee.pubkeys
    return exited_validator_index


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_with_participating_exited_member(spec: Any, state: Any) -> Iterator[Any]:
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    for _ in range(3):
        next_epoch_via_block(spec, state)
    committee_indices: List[int] = compute_committee_indices(state)
    rng: random.Random = random.Random(1010)
    exited_index: int = _exit_validator_from_committee_and_transition_state(
        spec, state, committee_indices, rng, lambda v: v.exit_epoch
    )
    current_epoch: int = spec.get_current_epoch(state)
    assert current_epoch < state.validators[exited_index].withdrawable_epoch
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, committee_indices, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block)


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_with_nonparticipating_exited_member(spec: Any, state: Any) -> Iterator[Any]:
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    for _ in range(3):
        next_epoch_via_block(spec, state)
    committee_indices: List[int] = compute_committee_indices(state)
    rng: random.Random = random.Random(1010)
    exited_index: int = _exit_validator_from_committee_and_transition_state(
        spec, state, committee_indices, rng, lambda v: v.exit_epoch
    )
    exited_pubkey: Any = state.validators[exited_index].pubkey
    current_epoch: int = spec.get_current_epoch(state)
    assert current_epoch < state.validators[exited_index].withdrawable_epoch
    exited_committee_index: int = state.current_sync_committee.pubkeys.index(exited_pubkey)
    block = build_empty_block_for_next_slot(spec, state)
    committee_bits: List[bool] = [i != exited_committee_index for i in committee_indices]
    filtered_committee_indices: List[int] = [index for index in committee_indices if index != exited_committee_index]
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=committee_bits,
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, filtered_committee_indices, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block)


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_with_participating_withdrawable_member(spec: Any, state: Any) -> Iterator[Any]:
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    for _ in range(3):
        next_epoch_via_block(spec, state)
    committee_indices: List[int] = compute_committee_indices(state)
    rng: random.Random = random.Random(1010)
    exited_index: int = _exit_validator_from_committee_and_transition_state(
        spec, state, committee_indices, rng, lambda v: v.withdrawable_epoch + 1
    )
    current_epoch: int = spec.get_current_epoch(state)
    assert current_epoch > state.validators[exited_index].withdrawable_epoch
    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, committee_indices, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block)


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_with_nonparticipating_withdrawable_member(spec: Any, state: Any) -> Iterator[Any]:
    state.slot += spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    for _ in range(3):
        next_epoch_via_block(spec, state)
    committee_indices: List[int] = compute_committee_indices(state)
    rng: random.Random = random.Random(1010)
    exited_index: int = _exit_validator_from_committee_and_transition_state(
        spec, state, committee_indices, rng, lambda v: v.withdrawable_epoch + 1
    )
    exited_pubkey: Any = state.validators[exited_index].pubkey
    current_epoch: int = spec.get_current_epoch(state)
    assert current_epoch > state.validators[exited_index].withdrawable_epoch
    target_committee_index: int = state.current_sync_committee.pubkeys.index(exited_pubkey)
    block = build_empty_block_for_next_slot(spec, state)
    committee_bits: List[bool] = [i != target_committee_index for i in committee_indices]
    filtered_committee_indices: List[int] = [index for index in committee_indices if index != target_committee_index]
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=committee_bits,
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec, state, block.slot - 1, filtered_committee_indices, block_root=block.parent_root
        ),
    )
    yield from run_sync_committee_processing(spec, state, block)
    
# End of test_sync_committee_with_nonparticipating_withdrawable_member


# No additional explanation.
