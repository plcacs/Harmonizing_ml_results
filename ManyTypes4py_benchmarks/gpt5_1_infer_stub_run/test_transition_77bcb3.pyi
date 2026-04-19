from typing import Any, Callable, Iterator, List, Literal, Optional, Protocol, Sequence, Tuple, Union


class Validator(Protocol):
    slashed: bool


class Checkpoint(Protocol):
    epoch: int


class State(Protocol):
    slot: int
    validators: Sequence[Validator]
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    historical_roots: List[bytes]


class BlockBody(Protocol):
    attestations: Sequence[Any]


class BlockMessage(Protocol):
    slot: int
    body: BlockBody


class SignedBlock(Protocol):
    message: BlockMessage


class Spec(Protocol):
    SLOTS_PER_EPOCH: int
    GENESIS_EPOCH: int

    def get_current_epoch(self, state: State) -> int: ...


YieldPre = Tuple[Literal["pre"], State]
YieldPost = Tuple[Literal["post"], State]
YieldBlocks = Tuple[Literal["blocks"], List[SignedBlock]]
YieldItem = Union[YieldPre, YieldPost, YieldBlocks]


def test_simple_transition(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_normal_transition(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_transition_randomized_state(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_transition_missing_first_post_block(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_transition_missing_last_pre_fork_block(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_transition_only_blocks_post_fork(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def _run_transition_test_with_attestations(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
    participation_fn: Optional[Callable[[int, int, Sequence[int]], Sequence[int]]] = ...,
    expect_finality: bool = ...,
) -> Iterator[YieldItem]: ...


def test_transition_with_finality(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_transition_with_random_three_quarters_participation(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_transition_with_random_half_participation(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_transition_with_no_attestations_until_after_fork(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...


def test_non_empty_historical_roots(
    state: State,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: Callable[[SignedBlock], SignedBlock],
    post_tag: Callable[[SignedBlock], SignedBlock],
) -> Iterator[YieldItem]: ...