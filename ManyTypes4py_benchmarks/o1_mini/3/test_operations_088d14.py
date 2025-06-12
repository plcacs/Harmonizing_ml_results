from typing import Generator
from eth2spec.test.context import ForkMeta, always_bls, with_fork_metas, with_presets
from eth2spec.test.helpers.constants import ALL_PRE_POST_FORKS, MINIMAL
from eth2spec.test.helpers.fork_transition import OperationType, run_transition_with_operation
from eth2spec.test.helpers.state import BeaconState
from eth2spec.test.helpers.spec import Spec

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_proposer_slashing_right_after_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create an attester slashing right *after* the transition
    """
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.PROPOSER_SLASHING,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_proposer_slashing_right_before_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create an attester slashing right *before* the transition
    """
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.PROPOSER_SLASHING,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_attester_slashing_right_after_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create an attester slashing right *after* the transition
    """
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.ATTESTER_SLASHING,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_attester_slashing_right_before_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create an attester slashing right *before* the transition
    """
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.ATTESTER_SLASHING,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_deposit_right_after_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create a deposit right *after* the transition
    """
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.DEPOSIT,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_deposit_right_before_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create a deposit right *before* the transition
    """
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.DEPOSIT,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=66) for pre, post in ALL_PRE_POST_FORKS])
@with_presets([MINIMAL], reason='too slow')
def test_transition_with_voluntary_exit_right_after_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create a voluntary exit right *after* the transition.
    fork_epoch=66 because minimal preset `SHARD_COMMITTEE_PERIOD` is 64 epochs.
    """
    state.slot = spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.VOLUNTARY_EXIT,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH
    )

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=66) for pre, post in ALL_PRE_POST_FORKS])
@with_presets([MINIMAL], reason='too slow')
def test_transition_with_voluntary_exit_right_before_fork(
    state: BeaconState,
    fork_epoch: int,
    spec: Spec,
    post_spec: Spec,
    pre_tag: str,
    post_tag: str
) -> Generator[None, None, None]:
    """
    Create a voluntary exit right *before* the transition.
    fork_epoch=66 because minimal preset `SHARD_COMMITTEE_PERIOD` is 64 epochs.
    """
    state.slot = spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    yield from run_transition_with_operation(
        state,
        fork_epoch,
        spec,
        post_spec,
        pre_tag,
        post_tag,
        operation_type=OperationType.VOLUNTARY_EXIT,
        operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1
    )
