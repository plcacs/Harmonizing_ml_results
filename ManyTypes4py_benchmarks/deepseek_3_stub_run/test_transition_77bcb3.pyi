import random
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple

from eth2spec.test.context import ForkMeta
from eth2spec.test.helpers.attestations import next_slots_with_attestations
from eth2spec.test.helpers.constants import ALL_PRE_POST_FORKS
from eth2spec.test.helpers.fork_transition import (
    do_fork,
    no_blocks,
    only_at,
    skip_slots,
    state_transition_across_slots,
    transition_to_next_epoch_and_append_blocks,
    transition_until_fork,
)
from eth2spec.test.helpers.random import randomize_state
from eth2spec.test.helpers.state import next_epoch_via_signed_block

def test_simple_transition(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_normal_transition(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_randomized_state(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_missing_first_post_block(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_missing_last_pre_fork_block(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_only_blocks_post_fork(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def _run_transition_test_with_attestations(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
    participation_fn: Optional[Callable[[int, int, List[int]], List[int]]] = None,
    expect_finality: bool = True,
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_finality(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_random_three_quarters_participation(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_random_half_participation(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_transition_with_no_attestations_until_after_fork(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...

def test_non_empty_historical_roots(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[Tuple[str, Any], None, None]: ...