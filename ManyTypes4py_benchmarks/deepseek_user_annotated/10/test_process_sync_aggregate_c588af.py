import random
from typing import Any, Callable, Generator, List, Set, Tuple, TypeVar, cast
from eth2spec.test.helpers.block import (
    build_empty_block_for_next_slot,
)
from eth2spec.test.helpers.state import (
    state_transition_and_sign_block,
    transition_to,
    next_epoch_via_block,
)
from eth2spec.test.helpers.constants import (
    MAINNET, MINIMAL,
)
from eth2spec.test.helpers.sync_committee import (
    compute_aggregate_sync_committee_signature,
    compute_committee_indices,
    run_sync_committee_processing,
    run_successful_sync_committee_test,
)
from eth2spec.test.helpers.voluntary_exits import (
    get_unslashed_exited_validators,
)
from eth2spec.test.context import (
    with_altair_and_later,
    with_presets,
    spec_state_test,
    always_bls,
    single_phase,
    with_custom_state,
    spec_test,
    default_balances_electra,
    default_activation_threshold,
)

T = TypeVar('T')

@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_bad_domain(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)

    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec,
            state,
            block.slot - 1,
            committee_indices,  # full committee signs
            block_root=block.parent_root,
            domain_type=spec.DOMAIN_BEACON_ATTESTER,  # Incorrect domain
        )
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_missing_participant(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)
    rng = random.Random(2020)
    random_participant = rng.choice(committee_indices)

    block = build_empty_block_for_next_slot(spec, state)
    # Exclude one participant whose signature was included.
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[index != random_participant for index in committee_indices],
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec,
            state,
            block.slot - 1,
            committee_indices,  # full committee signs
            block_root=block.parent_root,
        )
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_no_participants(spec: Any, state: Any) -> Generator[None, None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    # No participants is an allowed case, but needs a specific signature, not the full-zeroed signature.
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[False] * len(block.body.sync_aggregate.sync_committee_bits),
        sync_committee_signature=b'\x00' * 96
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_infinite_signature_with_all_participants(spec: Any, state: Any) -> Generator[None, None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    # Include all participants, try the special-case signature for no-participants
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(block.body.sync_aggregate.sync_committee_bits),
        sync_committee_signature=spec.G2_POINT_AT_INFINITY
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_infinite_signature_with_single_participant(spec: Any, state: Any) -> Generator[None, None, None]:
    block = build_empty_block_for_next_slot(spec, state)
    # Try include a single participant with the special-case signature for no-participants.
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] + ([False] * (len(block.body.sync_aggregate.sync_committee_bits) - 1)),
        sync_committee_signature=spec.G2_POINT_AT_INFINITY
    )
    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_extra_participant(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)
    rng = random.Random(3030)
    random_participant = rng.choice(committee_indices)

    block = build_empty_block_for_next_slot(spec, state)
    # Exclude one signature even though the block claims the entire committee participated.
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec,
            state,
            block.slot - 1,
            [index for index in committee_indices if index != random_participant],
            block_root=block.parent_root,
        )
    )

    yield from run_sync_committee_processing(spec, state, block, expect_exception=True)


def is_duplicate_sync_committee(committee_indices: List[int]) -> bool:
    dup = {v for v in committee_indices if committee_indices.count(v) > 1}
    return len(dup) > 0


@with_altair_and_later
@with_presets([MINIMAL], reason="to create nonduplicate committee")
@spec_test
@with_custom_state(balances_fn=default_balances_electra, threshold_fn=default_activation_threshold)
@single_phase
def test_sync_committee_rewards_nonduplicate_committee(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)

    # Preconditions of this test case
    assert not is_duplicate_sync_committee(committee_indices)

    committee_size = len(committee_indices)
    committee_bits = [True] * committee_size

    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@with_presets([MAINNET], reason="to create duplicate committee")
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_no_participation(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)

    # Preconditions of this test case
    assert is_duplicate_sync_committee(committee_indices)

    committee_size = len(committee_indices)
    committee_bits = [False] * committee_size

    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@with_presets([MAINNET], reason="to create duplicate committee")
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_half_participation(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)

    # Preconditions of this test case
    assert is_duplicate_sync_committee(committee_indices)

    committee_size = len(committee_indices)
    committee_bits = [True] * (committee_size // 2) + [False] * (committee_size // 2)
    assert len(committee_bits) == committee_size

    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@with_presets([MAINNET], reason="to create duplicate committee")
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_full_participation(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)

    # Preconditions of this test case
    assert is_duplicate_sync_committee(committee_indices)

    committee_size = len(committee_indices)
    committee_bits = [True] * committee_size

    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


def _run_sync_committee_selected_twice(
        spec: Any, 
        state: Any,
        pre_balance: int, 
        participate_first_position: bool, 
        participate_second_position: bool,
        skip_reward_validation: bool = False) -> Generator[None, None, int]:
    committee_indices = compute_committee_indices(state)

    # Preconditions of this test case
    assert is_duplicate_sync_committee(committee_indices)

    committee_size = len(committee_indices)
    committee_bits = [False] * committee_size

    # Find duplicate indices that get selected twice
    dup = {v for v in committee_indices if committee_indices.count(v) == 2}
    assert len(dup) > 0
    validator_index = dup.pop()
    positions = [i for i, v in enumerate(committee_indices) if v == validator_index]
    committee_bits[positions[0]] = participate_first_position
    committee_bits[positions[1]] = participate_second_position

    # Set validator's balance
    state.balances[validator_index] = pre_balance
    state.validators[validator_index].effective_balance = min(
        pre_balance - pre_balance % spec.EFFECTIVE_BALANCE_INCREMENT,
        spec.MAX_EFFECTIVE_BALANCE,
    )

    yield from run_successful_sync_committee_test(
        spec, state, committee_indices, committee_bits,
        skip_reward_validation=skip_reward_validation)

    return validator_index


@with_altair_and_later
@with_presets([MAINNET], reason="to create duplicate committee")
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_zero_balance_only_participate_first_one(spec: Any, state: Any) -> Generator[None, None, None]:
    validator_index = yield from _run_sync_committee_selected_twice(
        spec,
        state,
        pre_balance=0,
        participate_first_position=True,
        participate_second_position=False,
    )

    # The validator gets reward first (balance > 0) and then gets the same amount of penalty (balance == 0)
    assert state.balances[validator_index] == 0


@with_altair_and_later
@with_presets([MAINNET], reason="to create duplicate committee")
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_zero_balance_only_participate_second_one(spec: Any, state: Any) -> Generator[None, None, None]:
    # Skip `validate_sync_committee_rewards` because it doesn't handle the balance computation order
    # inside the for loop
    validator_index = yield from _run_sync_committee_selected_twice(
        spec,
        state,
        pre_balance=0,
        participate_first_position=False,
        participate_second_position=True,
        skip_reward_validation=True,
    )

    # The validator gets penalty first (balance is still 0) and then gets reward (balance > 0)
    assert state.balances[validator_index] > 0


@with_altair_and_later
@with_presets([MAINNET], reason="to create duplicate committee")
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_max_effective_balance_only_participate_first_one(spec: Any, state: Any) -> Generator[None, None, None]:
    validator_index = yield from _run_sync_committee_selected_twice(
        spec,
        state,
        pre_balance=spec.MAX_EFFECTIVE_BALANCE,
        participate_first_position=True,
        participate_second_position=False,
    )

    assert state.balances[validator_index] == spec.MAX_EFFECTIVE_BALANCE


@with_altair_and_later
@with_presets([MAINNET], reason="to create duplicate committee")
@spec_state_test
def test_sync_committee_rewards_duplicate_committee_max_effective_balance_only_participate_second_one(spec: Any, state: Any) -> Generator[None, None, None]:
    validator_index = yield from _run_sync_committee_selected_twice(
        spec,
        state,
        pre_balance=spec.MAX_EFFECTIVE_BALANCE,
        participate_first_position=False,
        participate_second_position=True,
    )

    assert state.balances[validator_index] == spec.MAX_EFFECTIVE_BALANCE


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_rewards_not_full_participants(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)
    rng = random.Random(1010)
    committee_bits = [rng.choice([True, False]) for _ in committee_indices]

    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@spec_state_test
@always_bls
def test_sync_committee_rewards_empty_participants(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)
    committee_bits = [False for _ in committee_indices]

    yield from run_successful_sync_committee_test(spec, state, committee_indices, committee_bits)


@with_altair_and_later
@spec_state_test
@always_bls
def test_invalid_signature_past_block(spec: Any, state: Any) -> Generator[None, None, None]:
    committee_indices = compute_committee_indices(state)

    for _ in range(2):
        # NOTE: need to transition twice to move beyond the degenerate case at genesis
        block = build_empty_block_for_next_slot(spec, state)
        # Valid sync committee signature here...
        block.body.sync_aggregate = spec.SyncAggregate(
            sync_committee_bits=[True] * len(committee_indices),
            sync_committee_signature=compute_aggregate_sync_committee_signature(
                spec,
                state,
                block.slot - 1,
                committee_indices,
                block_root=block.parent_root,
            )
        )

        state_transition_and_sign_block(spec, state, block)

    invalid_block = build_empty_block_for_next_slot(spec, state)
    # Invalid signature from a slot other than the previous
    invalid_block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec,
            state,
            invalid_block.slot - 2,
            committee_indices,
        )
    )

    yield from run_sync_committee_processing(spec, state, invalid_block, expect_exception=True)


@with_altair_and_later
@with_presets([MINIMAL], reason="to produce different committee sets")
@spec_state_test
@always_bls
def test_invalid_signature_previous_committee(spec: Any, state: Any) -> Generator[None, None, None]:
    # NOTE: the `state` provided is at genesis and the process to select
    # sync committees currently returns the same committee for the first and second
    # periods at genesis.
    # To get a distinct committee so we can generate an "old" signature, we need to advance
    # 2 EPOCHS_PER_SYNC_COMMITTEE_PERIOD periods.
    current_epoch = spec.get_current_epoch(state)
    old_sync_committee = state.next_sync_committee

    epoch_in_future_sync_committee_period = current_epoch + 2 * spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD
    slot_in_future_sync_committee_period = epoch_in_future_sync_committee_period * spec.SLOTS_PER_EPOCH
    transition_to(spec, state, slot_in_future_sync_committee_period)

    # Use the previous sync committee to produce the signature.
    # Ensure that the pubkey sets are different.
    assert set(old_sync_committee.pubkeys) != set(state.current_sync_committee.pubkeys)
    committee_indices = compute_committee_indices(state, old_sync_committee)

    block = build_empty_block_for_next_slot(spec, state)
    block.body.sync_aggregate = spec.SyncAggregate(
        sync_committee_bits=[True] * len(committee_indices),
        sync_committee_signature=compute_aggregate_sync_committee_signature(
            spec,
            state,
            block.slot - 1,
            committee_indices,
            block_root=block.parent_root,
        )
    )

    yield from run_sync_