from eth2spec.test.context import ForkMeta, always_bls, with_fork_metas, with_presets
from eth2spec.test.helpers.constants import ALL_PRE_POST_FORKS, MINIMAL
from eth2spec.test.helpers.fork_transition import OperationType, run_transition_with_operation

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_proposer_slashing_right_after_fork(state: Union[dict, list[str], str, None], fork_epoch: Union[dict, list[str], str, None], spec: Union[dict, list[str], str, None], post_spec: Union[dict, list[str], str, None], pre_tag: Union[dict, list[str], str, None], post_tag: Union[dict, list[str], str, None]) -> typing.Generator:
    """
    Create an attester slashing right *after* the transition
    """
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.PROPOSER_SLASHING, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_proposer_slashing_right_before_fork(state: Union[dict, str, None, bool], fork_epoch: Union[dict, str, None, bool], spec: Union[dict, str, None, bool], post_spec: Union[dict, str, None, bool], pre_tag: Union[dict, str, None, bool], post_tag: Union[dict, str, None, bool]) -> typing.Generator:
    """
    Create an attester slashing right *before* the transition
    """
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.PROPOSER_SLASHING, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_attester_slashing_right_after_fork(state: Union[str, dict], fork_epoch: Union[str, dict], spec: Union[str, dict], post_spec: Union[str, dict], pre_tag: Union[str, dict], post_tag: Union[str, dict]) -> typing.Generator:
    """
    Create an attester slashing right *after* the transition
    """
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.ATTESTER_SLASHING, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
@always_bls
def test_transition_with_attester_slashing_right_before_fork(state: Union[dict, list[str], str, None], fork_epoch: Union[dict, list[str], str, None], spec: Union[dict, list[str], str, None], post_spec: Union[dict, list[str], str, None], pre_tag: Union[dict, list[str], str, None], post_tag: Union[dict, list[str], str, None]) -> typing.Generator:
    """
    Create an attester slashing right *after* the transition
    """
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.ATTESTER_SLASHING, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_deposit_right_after_fork(state: Union[dict[str, typing.Any], list[str], dict], fork_epoch: Union[dict[str, typing.Any], list[str], dict], spec: Union[dict[str, typing.Any], list[str], dict], post_spec: Union[dict[str, typing.Any], list[str], dict], pre_tag: Union[dict[str, typing.Any], list[str], dict], post_tag: Union[dict[str, typing.Any], list[str], dict]) -> typing.Generator:
    """
    Create a deposit right *after* the transition
    """
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.DEPOSIT, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=2) for pre, post in ALL_PRE_POST_FORKS])
def test_transition_with_deposit_right_before_fork(state: Union[list[str], typing.Callable, dict[str, typing.Any]], fork_epoch: Union[list[str], typing.Callable, dict[str, typing.Any]], spec: Union[list[str], typing.Callable, dict[str, typing.Any]], post_spec: Union[list[str], typing.Callable, dict[str, typing.Any]], pre_tag: Union[list[str], typing.Callable, dict[str, typing.Any]], post_tag: Union[list[str], typing.Callable, dict[str, typing.Any]]) -> typing.Generator:
    """
    Create a deposit right *before* the transition
    """
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.DEPOSIT, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=66) for pre, post in ALL_PRE_POST_FORKS])
@with_presets([MINIMAL], reason='too slow')
def test_transition_with_voluntary_exit_right_after_fork(state: Union[dict, None], fork_epoch: Union[list, typing.Callable[str, bool]], spec: Union[dict, None], post_spec: Union[list, typing.Callable[str, bool]], pre_tag: Union[list, typing.Callable[str, bool]], post_tag: Union[list, typing.Callable[str, bool]]) -> typing.Generator:
    """
    Create a voluntary exit right *after* the transition.
    fork_epoch=66 because minimal preset `SHARD_COMMITTEE_PERIOD` is 64 epochs.
    """
    state.slot = spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.VOLUNTARY_EXIT, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH)

@with_fork_metas([ForkMeta(pre_fork_name=pre, post_fork_name=post, fork_epoch=66) for pre, post in ALL_PRE_POST_FORKS])
@with_presets([MINIMAL], reason='too slow')
def test_transition_with_voluntary_exit_right_before_fork(state: Union[dict, None, str], fork_epoch: Union[list, typing.Callable[str, bool]], spec: Union[dict, None, str], post_spec: Union[list, typing.Callable[str, bool]], pre_tag: Union[list, typing.Callable[str, bool]], post_tag: Union[list, typing.Callable[str, bool]]) -> typing.Generator:
    """
    Create a voluntary exit right *before* the transition.
    fork_epoch=66 because minimal preset `SHARD_COMMITTEE_PERIOD` is 64 epochs.
    """
    state.slot = spec.config.SHARD_COMMITTEE_PERIOD * spec.SLOTS_PER_EPOCH
    yield from run_transition_with_operation(state, fork_epoch, spec, post_spec, pre_tag, post_tag, operation_type=OperationType.VOLUNTARY_EXIT, operation_at_slot=fork_epoch * spec.SLOTS_PER_EPOCH - 1)