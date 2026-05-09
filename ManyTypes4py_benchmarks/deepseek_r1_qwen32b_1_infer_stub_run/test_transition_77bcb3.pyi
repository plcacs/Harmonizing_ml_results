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
from eth2spec import spec

def test_simple_transition(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_normal_transition(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_transition_randomized_state(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_transition_missing_first_post_block(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_transition_missing_last_pre_fork_block(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_transition_only_blocks_post_fork(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def _run_transition_test_with_attestations(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
    participation_fn: Callable[[int, int, List[int]], List[int]] = None,
    expect_finality: bool = True,
) -> Generator[Tuple[str, spec.BeaconState], None, None]:
    ...

def test_transition_with_finality(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_transition_with_random_three_quarters_participation(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_transition_with_random_half_participation(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_transition_with_no_attestations_until_after_fork(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...

def test_non_empty_historical_roots(
    state: spec.BeaconState,
    fork_epoch: int,
    spec: spec.Spec,
    post_spec: spec.Spec,
    pre_tag: spec.Spec,
    post_tag: spec.Spec,
) -> None:
    ...