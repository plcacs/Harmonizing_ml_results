```python
import random
from typing import Any, Iterator, Tuple, Union, Callable, Optional, List

from eth2spec.test.context import ForkMeta
from eth2spec.test.helpers.random import randomize_state
from eth2spec.test.helpers.constants import ALL_PRE_POST_FORKS
from eth2spec.test.helpers.state import next_epoch_via_signed_block
from eth2spec.test.helpers.attestations import next_slots_with_attestations
from eth2spec.test.helpers.fork_transition import do_fork, no_blocks, only_at, skip_slots, state_transition_across_slots, transition_to_next_epoch_and_append_blocks, transition_until_fork

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_simple_transition(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_normal_transition(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=8) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_randomized_state(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_first_post_block(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_last_pre_fork_block(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_only_blocks_post_fork(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

def _run_transition_test_with_attestations(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any, participation_fn: Optional[Callable[..., Any]] = ..., expect_finality: bool = ...) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_finality(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_three_quarters_participation(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_half_participation(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=3) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_no_attestations_until_after_fork(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_non_empty_historical_roots(state: Any, fork_epoch: Any, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator[Tuple[str, Any]]: ...
```