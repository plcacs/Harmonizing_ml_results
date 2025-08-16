from typing import Iterator

def test_simple_transition(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_normal_transition(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_randomized_state(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_missing_first_post_block(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_missing_last_pre_fork_block(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_only_blocks_post_fork(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_with_finality(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_with_random_three_quarters_participation(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_with_random_half_participation(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_transition_with_no_attestations_until_after_fork(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
def test_non_empty_historical_roots(state, fork_epoch, spec, post_spec, pre_tag, post_tag) -> Iterator[str, Any]:
