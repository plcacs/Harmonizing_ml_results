from typing import Any, Callable, Iterator, List, Optional, Tuple
import random
from eth2spec.test.context import ForkMeta, with_fork_metas
from eth2spec.test.helpers.random import randomize_state
from eth2spec.test.helpers.constants import ALL_PRE_POST_FORKS
from eth2spec.test.helpers.state import next_epoch_via_signed_block
from eth2spec.test.helpers.attestations import next_slots_with_attestations
from eth2spec.test.helpers.fork_transition import (
    do_fork,
    no_blocks,
    only_at,
    skip_slots,
    state_transition_across_slots,
    transition_to_next_epoch_and_append_blocks,
    transition_until_fork,
)

# Assume Block and State are not explicitly defined here, so we use Any.
Block = Any
State = Any
Spec = Any

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_simple_transition(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    transition_until_fork(spec, state, fork_epoch)
    assert spec.get_current_epoch(state) < fork_epoch
    yield ('pre', state)
    blocks: List[Block] = []
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks, only_last_block=True)
    yield ('blocks', blocks)
    yield ('post', state)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_normal_transition(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Transition from the initial ``state`` to the epoch after the ``fork_epoch``,
    producing blocks for every slot along the way.
    """
    yield ('pre', state)
    assert spec.get_current_epoch(state) < fork_epoch
    to_slot: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    blocks: List[Block] = []
    blocks.extend([pre_tag(block) for block in state_transition_across_slots(spec, state, to_slot)])
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks)
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks: List[int] = [block.message.slot for block in blocks]
    assert len(set(slots_with_blocks)) == len(slots_with_blocks)
    assert set(range(1, state.slot + 1)) == set(slots_with_blocks)
    yield ('blocks', blocks)
    yield ('post', state)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=8) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_randomized_state(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    randomize_state(spec, state)
    transition_until_fork(spec, state, fork_epoch)
    assert spec.get_current_epoch(state) < fork_epoch
    yield ('pre', state)
    blocks: List[Block] = []
    state, _ = do_fork(state, spec, post_spec, fork_epoch, with_block=False)
    slashed_indices: List[int] = [index for index, validator in enumerate(state.validators) if validator.slashed]
    transition_to_next_epoch_and_append_blocks(
        post_spec, state, post_tag, blocks, only_last_block=True, ignoring_proposers=slashed_indices
    )
    yield ('blocks', blocks)
    yield ('post', state)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_first_post_block(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Transition from the initial ``state`` to the epoch after the ``fork_epoch``,
    producing blocks for every slot along the way except for the first block
    of the new fork.
    """
    yield ('pre', state)
    assert spec.get_current_epoch(state) < fork_epoch
    to_slot: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    blocks: List[Block] = []
    blocks.extend([pre_tag(block) for block in state_transition_across_slots(spec, state, to_slot)])
    state, _ = do_fork(state, spec, post_spec, fork_epoch, with_block=False)
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks)
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks: List[int] = [block.message.slot for block in blocks]
    assert len(set(slots_with_blocks)) == len(slots_with_blocks)
    expected_slots = set(range(1, state.slot + 1)).difference({fork_epoch * spec.SLOTS_PER_EPOCH})
    assert expected_slots == set(slots_with_blocks)
    yield ('blocks', blocks)
    yield ('post', state)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_last_pre_fork_block(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Transition from the initial ``state`` to the epoch after the ``fork_epoch``,
    producing blocks for every slot along the way except for the last block
    of the old fork.
    """
    yield ('pre', state)
    assert spec.get_current_epoch(state) < fork_epoch
    last_slot_of_pre_fork: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    to_slot: int = last_slot_of_pre_fork
    blocks: List[Block] = []
    blocks.extend([
        pre_tag(block)
        for block in state_transition_across_slots(
            spec, state, to_slot, block_filter=skip_slots(last_slot_of_pre_fork)
        )
    ])
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks)
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks = [block.message.slot for block in blocks]
    assert len(set(slots_with_blocks)) == len(slots_with_blocks)
    expected_slots = set(range(1, state.slot + 1)).difference({last_slot_of_pre_fork})
    assert expected_slots == set(slots_with_blocks)
    yield ('blocks', blocks)
    yield ('post', state)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_only_blocks_post_fork(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Transition from the initial ``state`` to the epoch after the ``fork_epoch``,
    skipping blocks for every slot along the way except for the first block
    in the ending epoch.
    """
    yield ('pre', state)
    assert spec.get_current_epoch(state) < fork_epoch
    last_slot_of_pre_fork: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    to_slot: int = last_slot_of_pre_fork
    blocks: List[Block] = []
    blocks.extend([
        pre_tag(block)
        for block in state_transition_across_slots(spec, state, to_slot, block_filter=no_blocks)
    ])
    state, _ = do_fork(state, spec, post_spec, fork_epoch, with_block=False)
    to_slot = post_spec.SLOTS_PER_EPOCH + state.slot
    last_slot: int = (fork_epoch + 1) * post_spec.SLOTS_PER_EPOCH
    blocks.extend([
        post_tag(block)
        for block in state_transition_across_slots(post_spec, state, to_slot, block_filter=only_at(last_slot))
    ])
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks: List[int] = [block.message.slot for block in blocks]
    assert len(slots_with_blocks) == 1
    assert slots_with_blocks[0] == last_slot
    yield ('blocks', blocks)
    yield ('post', state)

def _run_transition_test_with_attestations(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
    participation_fn: Optional[Callable[[Any, Any, List[int]], List[int]]] = None,
    expect_finality: bool = True,
) -> Iterator[Tuple[str, Any]]:
    yield ('pre', state)
    current_epoch: int = spec.get_current_epoch(state)
    assert current_epoch < fork_epoch
    assert current_epoch == spec.GENESIS_EPOCH
    block: Block = next_epoch_via_signed_block(spec, state)
    fill_cur_epoch: bool = False
    fill_prev_epoch: bool = True
    blocks: List[Block] = [pre_tag(block)]
    current_epoch = spec.get_current_epoch(state)
    for _ in range(current_epoch, fork_epoch - 1):
        _, blocks_in_epoch, state = next_slots_with_attestations(
            spec, state, spec.SLOTS_PER_EPOCH, fill_cur_epoch, fill_prev_epoch, participation_fn=participation_fn
        )
        blocks.extend([pre_tag(block) for block in blocks_in_epoch])
    _, blocks_in_epoch, state = next_slots_with_attestations(
        spec, state, spec.SLOTS_PER_EPOCH - 1, fill_cur_epoch, fill_prev_epoch, participation_fn=participation_fn
    )
    blocks.extend([pre_tag(block) for block in blocks_in_epoch])
    assert spec.get_current_epoch(state) == fork_epoch - 1
    assert (state.slot + 1) % spec.SLOTS_PER_EPOCH == 0
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    for _ in range(4):
        _, blocks_in_epoch, state = next_slots_with_attestations(
            post_spec, state, post_spec.SLOTS_PER_EPOCH, fill_cur_epoch, fill_prev_epoch, participation_fn=participation_fn
        )
        blocks.extend([post_tag(block) for block in blocks_in_epoch])
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 4
    if expect_finality:
        assert state.current_justified_checkpoint.epoch == fork_epoch + 2
        assert state.finalized_checkpoint.epoch == fork_epoch
    else:
        assert state.current_justified_checkpoint.epoch == spec.GENESIS_EPOCH
        assert state.finalized_checkpoint.epoch == spec.GENESIS_EPOCH
    assert len(blocks) == (fork_epoch + 3) * post_spec.SLOTS_PER_EPOCH + 1
    assert len(blocks) == len(set(blocks))
    blocks_without_attestations: List[Block] = [block for block in blocks if len(block.message.body.attestations) == 0]
    assert len(blocks_without_attestations) == 2
    slots_without_attestations: List[int] = [b.message.slot for b in blocks_without_attestations]
    assert set(slots_without_attestations) == {spec.SLOTS_PER_EPOCH, fork_epoch * spec.SLOTS_PER_EPOCH}
    yield ('blocks', blocks)
    yield ('post', state)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_finality(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Transition from the initial ``state`` to the epoch after the ``fork_epoch``,
    including attestations so as to produce finality through the fork boundary.
    """
    yield from _run_transition_test_with_attestations(state, fork_epoch, spec, post_spec, pre_tag, post_tag)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_three_quarters_participation(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Transition from the initial ``state`` to the epoch after the ``fork_epoch``,
    including attestations so as to produce finality through the fork boundary.
    """
    rng = random.Random(1337)

    def _drop_random_quarter(_slot: Any, _index: Any, indices: List[int]) -> List[int]:
        committee_len: int = len(indices)
        assert committee_len >= 4
        filter_len: int = committee_len // 4
        participant_count: int = committee_len - filter_len
        return rng.sample(sorted(indices), participant_count)

    yield from _run_transition_test_with_attestations(
        state, fork_epoch, spec, post_spec, pre_tag, post_tag, participation_fn=_drop_random_quarter
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_half_participation(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    rng = random.Random(2020)

    def _drop_random_half(_slot: Any, _index: Any, indices: List[int]) -> List[int]:
        committee_len: int = len(indices)
        assert committee_len >= 2
        filter_len: int = committee_len // 2
        participant_count: int = committee_len - filter_len
        return rng.sample(sorted(indices), participant_count)

    yield from _run_transition_test_with_attestations(
        state, fork_epoch, spec, post_spec, pre_tag, post_tag, participation_fn=_drop_random_half, expect_finality=False
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_no_attestations_until_after_fork(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Transition from the initial ``state`` to the ``fork_epoch`` with no attestations,
    then transition forward with enough attestations to finalize the fork epoch.
    """
    yield ('pre', state)
    assert spec.get_current_epoch(state) < fork_epoch
    to_slot: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    blocks: List[Block] = []
    blocks.extend([pre_tag(block) for block in state_transition_across_slots(spec, state, to_slot)])
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    block = next_epoch_via_signed_block(post_spec, state)
    blocks.append(post_tag(block))
    for _ in range(4):
        _, blocks_in_epoch, state = next_slots_with_attestations(post_spec, state, post_spec.SLOTS_PER_EPOCH, False, True)
        blocks.extend([post_tag(block) for block in blocks_in_epoch])
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 5
    assert state.current_justified_checkpoint.epoch == fork_epoch + 3
    assert state.finalized_checkpoint.epoch == fork_epoch + 1
    yield ('blocks', blocks)
    yield ('post', state)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_non_empty_historical_roots(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[Block], Block],
    post_tag: Callable[[Block], Block],
) -> Iterator[Tuple[str, Any]]:
    """
    Test with non-empty pre-state `state.historical_roots`.

    Since Capella froze `historical_roots`, Capella spec doesn't invoke `process_historical_roots_update` anymore.
    Therefore, we need to fill in `historical_roots` with non-empty value.
    """
    pre_historical_roots: List[bytes] = [b'V' * 32]
    state.historical_roots = pre_historical_roots
    transition_until_fork(spec, state, fork_epoch)
    assert spec.get_current_epoch(state) < fork_epoch
    assert len(state.historical_roots) > 0
    yield ('pre', state)
    blocks: List[Block] = []
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks, only_last_block=True)
    yield ('blocks', blocks)
    yield ('post', state)
    assert len(state.historical_roots) > 0
    assert state.historical_roots == pre_historical_roots