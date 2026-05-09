from eth2spec.test.context import ForkMeta, with_fork_metas
from eth2spec.test.helpers.random import randomize_state
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
from typing import (
    Generator,
    List,
    Callable,
    Optional,
    Tuple,
    Union,
)
import random

T = Union[bytes, str]

def test_simple_transition(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_normal_transition(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_randomized_state(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_missing_first_post_block(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_missing_last_pre_fork_block(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_only_blocks_post_fork(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def _run_transition_test_with_attestations(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
    participation_fn: Optional[Callable[[int, int, List[int]], List[int]]] = None,
    expect_finality: bool = True,
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_with_finality(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_with_random_three_quarters_participation(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_with_random_half_participation(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_transition_with_no_attestations_until_after_fork(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...

def test_non_empty_historical_roots(
    state: object,
    fork_epoch: int,
    spec: object,
    post_spec: object,
    pre_tag: Callable[[object], T],
    post_tag: Callable[[object], T],
) -> Generator[Tuple[str, object], None, None]:
    ...