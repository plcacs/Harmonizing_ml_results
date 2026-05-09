from typing import Any, Generator, List, Optional, Tuple, Callable, Set
from eth2spec.test.context import ForkMeta

# Assuming the following types based on the usage in the module
# state: The state of the beacon chain
# spec: The specification object
# block: A block object
# pre_tag/post_tag: Functions that wrap/tag blocks for different forks

State = Any
Spec = Any
Block = Any
TagFn = Callable[[Block], Block]
ParticipationFn = Callable[[int, int, List[int]], List[int]]

def test_simple_transition(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_normal_transition(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_randomized_state(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_missing_first_post_block(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_missing_last_pre_fork_block(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_only_blocks_post_fork(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def _run_transition_test_with_attestations(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
    participation_fn: Optional[ParticipationFn] = None,
    expect_finality: bool = True,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_finality(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_random_three_quarters_participation(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_random_half_participation(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_no_attestations_until_after_fork(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_non_empty_historical_roots(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: TagFn,
    post_tag: TagFn,
) -> Generator[Tuple[str, Any], None, None]: ...