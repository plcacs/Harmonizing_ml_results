import random
from typing import List, Tuple, Generator, Optional, Callable, Set
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
from eth2spec.phase0 import spec as phase0_spec
from eth2spec.altair import spec as altair_spec
from eth2spec.bellatrix import spec as bellatrix_spec
from eth2spec.capella import spec as capella_spec
from eth2spec.deneb import spec as deneb_spec

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_simple_transition(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    transition_until_fork(spec, state, fork_epoch)
    assert spec.get_current_epoch(state) < fork_epoch
    yield "pre", state
    blocks: List[phase0_spec.SignedBeaconBlock] = []
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks, only_last_block=True)
    yield "blocks", blocks
    yield "post", state

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_normal_transition(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    yield "pre", state
    assert spec.get_current_epoch(state) < fork_epoch
    to_slot: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    blocks: List[phase0_spec.SignedBeaconBlock] = []
    blocks.extend([pre_tag(block) for block in state_transition_across_slots(spec, state, to_slot)])
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks)
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks: List[int] = [block.message.slot for block in blocks]
    assert len(set(slots_with_blocks)) == len(slots_with_blocks)
    assert set(range(1, state.slot + 1)) == set(slots_with_blocks)
    yield "blocks", blocks
    yield "post", state

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=8) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_randomized_state(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    randomize_state(spec, state)
    transition_until_fork(spec, state, fork_epoch)
    assert spec.get_current_epoch(state) < fork_epoch
    yield "pre", state
    blocks: List[phase0_spec.SignedBeaconBlock] = []
    state, _ = do_fork(state, spec, post_spec, fork_epoch, with_block=False)
    slashed_indices: List[int] = [index for index, validator in enumerate(state.validators) if validator.slashed]
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks, only_last_block=True, ignoring_proposers=slashed_indices)
    yield "blocks", blocks
    yield "post", state

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_first_post_block(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    yield "pre", state
    assert spec.get_current_epoch(state) < fork_epoch
    to_slot: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    blocks: List[phase0_spec.SignedBeaconBlock] = []
    blocks.extend([pre_tag(block) for block in state_transition_across_slots(spec, state, to_slot)])
    state, _ = do_fork(state, spec, post_spec, fork_epoch, with_block=False)
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks)
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks: List[int] = [block.message.slot for block in blocks]
    assert len(set(slots_with_blocks)) == len(slots_with_blocks)
    expected_slots: Set[int] = set(range(1, state.slot + 1)).difference(set([fork_epoch * spec.SLOTS_PER_EPOCH]))
    assert expected_slots == set(slots_with_blocks)
    yield "blocks", blocks
    yield "post", state

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_last_pre_fork_block(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    yield "pre", state
    assert spec.get_current_epoch(state) < fork_epoch
    last_slot_of_pre_fork: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    to_slot: int = last_slot_of_pre_fork
    blocks: List[phase0_spec.SignedBeaconBlock] = []
    blocks.extend([pre_tag(block) for block in state_transition_across_slots(spec, state, to_slot, block_filter=skip_slots(last_slot_of_pre_fork))])
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks)
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks: List[int] = [block.message.slot for block in blocks]
    assert len(set(slots_with_blocks)) == len(slots_with_blocks)
    expected_slots: Set[int] = set(range(1, state.slot + 1)).difference(set([last_slot_of_pre_fork]))
    assert expected_slots == set(slots_with_blocks)
    yield "blocks", blocks
    yield "post", state

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_only_blocks_post_fork(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    yield "pre", state
    assert spec.get_current_epoch(state) < fork_epoch
    last_slot_of_pre_fork: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    to_slot: int = last_slot_of_pre_fork
    blocks: List[phase0_spec.SignedBeaconBlock] = []
    blocks.extend([pre_tag(block) for block in state_transition_across_slots(spec, state, to_slot, block_filter=no_blocks)])
    state, _ = do_fork(state, spec, post_spec, fork_epoch, with_block=False)
    to_slot: int = post_spec.SLOTS_PER_EPOCH + state.slot
    last_slot: int = (fork_epoch + 1) * post_spec.SLOTS_PER_EPOCH
    blocks.extend([post_tag(block) for block in state_transition_across_slots(post_spec, state, to_slot, block_filter=only_at(last_slot))])
    assert state.slot % post_spec.SLOTS_PER_EPOCH == 0
    assert post_spec.get_current_epoch(state) == fork_epoch + 1
    slots_with_blocks: List[int] = [block.message.slot for block in blocks]
    assert len(slots_with_blocks) == 1
    assert slots_with_blocks[0] == last_slot
    yield "blocks", blocks
    yield "post", state

def _run_transition_test_with_attestations(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str, participation_fn: Optional[Callable[[int, int, List[int]], List[int]]] = None, expect_finality: bool = True) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    yield "pre", state
    current_epoch: int = spec.get_current_epoch(state)
    assert current_epoch < fork_epoch
    assert current_epoch == spec.GENESIS_EPOCH
    block: phase0_spec.SignedBeaconBlock = next_epoch_via_signed_block(spec, state)
    fill_cur_epoch: bool = False
    fill_prev_epoch: bool = True
    blocks: List[phase0_spec.SignedBeaconBlock] = [pre_tag(block)]
    current_epoch = spec.get_current_epoch(state)
    for _ in range(current_epoch, fork_epoch - 1):
        _, blocks_in_epoch, state = next_slots_with_attestations(spec, state, spec.SLOTS_PER_EPOCH, fill_cur_epoch, fill_prev_epoch, participation_fn=participation_fn)
        blocks.extend([pre_tag(block) for block in blocks_in_epoch])
    _, blocks_in_epoch, state = next_slots_with_attestations(spec, state, spec.SLOTS_PER_EPOCH - 1, fill_cur_epoch, fill_prev_epoch, participation_fn=participation_fn)
    blocks.extend([pre_tag(block) for block in blocks_in_epoch])
    assert spec.get_current_epoch(state) == fork_epoch - 1
    assert (state.slot + 1) % spec.SLOTS_PER_EPOCH == 0
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    for _ in range(4):
        _, blocks_in_epoch, state = next_slots_with_attestations(post_spec, state, post_spec.SLOTS_PER_EPOCH, fill_cur_epoch, fill_prev_epoch, participation_fn=participation_fn)
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
    blocks_without_attestations: List[phase0_spec.SignedBeaconBlock] = [block for block in blocks if len(block.message.body.attestations) == 0]
    assert len(blocks_without_attestations) == 2
    slots_without_attestations: List[int] = [b.message.slot for b in blocks_without_attestations]
    assert set(slots_without_attestations) == set([spec.SLOTS_PER_EPOCH, fork_epoch * spec.SLOTS_PER_EPOCH])
    yield "blocks", blocks
    yield "post", state

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_finality(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    yield from _run_transition_test_with_attestations(state, fork_epoch, spec, post_spec, pre_tag, post_tag)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_three_quarters_participation(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    rng: random.Random = random.Random(1337)
    def _drop_random_quarter(_slot: int, _index: int, indices: List[int]) -> List[int]:
        committee_len: int = len(indices)
        assert committee_len >= 4
        filter_len: int = committee_len // 4
        participant_count: int = committee_len - filter_len
        return rng.sample(sorted(indices), participant_count)
    yield from _run_transition_test_with_attestations(state, fork_epoch, spec, post_spec, pre_tag, post_tag, participation_fn=_drop_random_quarter)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_half_participation(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    rng: random.Random = random.Random(2020)
    def _drop_random_half(_slot: int, _index: int, indices: List[int]) -> List[int]:
        committee_len: int = len(indices)
        assert committee_len >= 2
        filter_len: int = committee_len // 2
        participant_count: int = committee_len - filter_len
        return rng.sample(sorted(indices), participant_count)
    yield from _run_transition_test_with_attestations(state, fork_epoch, spec, post_spec, pre_tag, post_tag, participation_fn=_drop_random_half, expect_finality=False)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_no_attestations_until_after_fork(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    yield "pre", state
    assert spec.get_current_epoch(state) < fork_epoch
    to_slot: int = fork_epoch * spec.SLOTS_PER_EPOCH - 1
    blocks: List[phase0_spec.SignedBeaconBlock] = []
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
    yield "blocks", blocks
    yield "post", state

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_non_empty_historical_roots(state: phase0_spec.BeaconState, fork_epoch: int, spec: phase0_spec.Spec, post_spec: phase0_spec.Spec, pre_tag: str, post_tag: str) -> Generator[Tuple[str, phase0_spec.BeaconState], None, None]:
    pre_historical_roots: List[bytes] = [b'\x56' * 32]
    state.historical_roots = pre_historical_roots
    transition_until_fork(spec, state, fork_epoch)
    assert spec.get_current_epoch(state) < fork_epoch
    assert len(state.historical_roots) > 0
    yield "pre", state
    blocks: List[phase0_spec.SignedBeaconBlock] = []
    state, block = do_fork(state, spec, post_spec, fork_epoch)
    blocks.append(post_tag(block))
    transition_to_next_epoch_and_append_blocks(post_spec, state, post_tag, blocks, only_last_block=True)
    yield "blocks", blocks
    yield "post", state
    assert len(state.historical_roots) > 0
    assert state.historical_roots == pre_historical_roots
