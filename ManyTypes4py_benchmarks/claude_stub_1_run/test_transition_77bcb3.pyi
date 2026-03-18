```pyi
from typing import Any, Callable, Generator, Optional

def test_simple_transition(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_normal_transition(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_randomized_state(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_missing_first_post_block(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_missing_last_pre_fork_block(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_only_blocks_post_fork(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def _run_transition_test_with_attestations(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
    participation_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
    expect_finality: bool = True,
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_with_finality(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_with_random_three_quarters_participation(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_with_random_half_participation(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_transition_with_no_attestations_until_after_fork(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...

def test_non_empty_historical_roots(
    state: Any,
    fork_epoch: int,
    spec: Any,
    post_spec: Any,
    pre_tag: Callable[[Any], Any],
    post_tag: Callable[[Any], Any],
) -> Generator[tuple[str, Any], None, None]: ...
```