from eth2spec.test.context import ForkMeta, always_bls, with_fork_metas, with_presets
from eth2spec.test.helpers.constants import ALL_PRE_POST_FORKS, MINIMAL
from eth2spec.test.helpers.fork_transition import OperationType, run_transition_with_operation
from typing import Iterator

def test_transition_with_proposer_slashing_right_after_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...

def test_transition_with_proposer_slashing_right_before_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...

def test_transition_with_attester_slashing_right_after_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...

def test_transition_with_attester_slashing_right_before_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...

def test_transition_with_deposit_right_after_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...

def test_transition_with_deposit_right_before_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...

def test_transition_with_voluntary_exit_right_after_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...

def test_transition_with_voluntary_exit_right_before_fork(state: Any, fork_epoch: int, spec: Any, post_spec: Any, pre_tag: Any, post_tag: Any) -> Iterator:
    ...
