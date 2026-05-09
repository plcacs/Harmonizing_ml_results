import random
from eth2spec.test.context import ForkMeta, with_fork_metas
from eth2spec.test.helpers.random import randomize_state
from eth2spec.test.helpers.constants import ALL_PRE_POST_FORKS
from eth2spec.test.helpers.state import next_epoch_via_signed_block
from eth2spec.test.helpers.attestations import next_slots_with_attestations
from eth2spec.test.helpers.fork_transition import do_fork, no_blocks, only_at, skip_slots, state_transition_across_slots, transition_to_next_epoch_and_append_blocks, transition_until_fork

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_simple_transition(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_normal_transition(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_randomized_state(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_first_post_block(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_missing_last_pre_fork_block(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_only_blocks_post_fork(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

def _run_transition_test_with_attestations(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable, participation_fn: callable = None, expect_finality: bool = True) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_finality(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_three_quarters_participation(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_random_half_participation(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_no_attestations_until_after_fork(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...

@with_fork_metas([ForkMeta(pre_fork_name: str, post_fork_name: str, fork_epoch: int) for pre, post in ALL_PRE_POST_FORKS])
def test_non_empty_historical_roots(state: dict, fork_epoch: int, spec: object, post_spec: object, pre_tag: callable, post_tag: callable) -> None:
    # ...
