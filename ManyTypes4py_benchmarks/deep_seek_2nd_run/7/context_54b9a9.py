import pytest
from copy import deepcopy
from dataclasses import dataclass
import importlib
from eth2spec.utils import bls
from .exceptions import SkippedTest
from .helpers.constants import PHASE0, ALTAIR, BELLATRIX, CAPELLA, DENEB, ELECTRA, FULU, WHISK, MINIMAL, ALL_PHASES, POST_FORK_OF, ALLOWED_TEST_RUNNER_FORKS, LIGHT_CLIENT_TESTING_FORKS
from .helpers.forks import is_post_fork, is_post_electra
from .helpers.genesis import create_genesis_state
from .helpers.typing import Spec, SpecForks
from .helpers.specs import spec_targets
from .utils import vector_test, with_meta_tags
from random import Random
from typing import Any, Callable, Sequence, Dict, List, Optional, TypeVar, Union, Tuple, Set, Generator, Iterator
from lru import LRU
from types import ModuleType

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

DEFAULT_TEST_PRESET = MINIMAL
DEFAULT_PYTEST_FORKS = ALL_PHASES

@dataclass(frozen=True)
class ForkMeta:
    pre_fork_name: str
    post_fork_name: str
    fork_epoch: Optional[int] = None

def _prepare_state(balances_fn: Callable[[Spec], List[int]], threshold_fn: Callable[[Spec], int], spec: Spec, phases: Dict[str, Spec]) -> Any:
    balances = balances_fn(spec)
    activation_threshold = threshold_fn(spec)
    state = create_genesis_state(spec=spec, validator_balances=balances, activation_threshold=activation_threshold)
    return state

_custom_state_cache_dict: LRU = LRU(size=10)

def with_custom_state(balances_fn: Callable[[Spec], List[int]], threshold_fn: Callable[[Spec], int]) -> Callable[[F], F]:

    def deco(fn: F) -> F:

        def entry(*args: Any, spec: Spec, phases: Dict[str, Spec], **kw: Any) -> Any:
            key = (spec.fork, spec.config.__hash__(), spec.__file__, balances_fn, threshold_fn)
            global _custom_state_cache_dict
            if key not in _custom_state_cache_dict:
                state = _prepare_state(balances_fn, threshold_fn, spec, phases)
                _custom_state_cache_dict[key] = state.get_backing()
            state = spec.BeaconState(backing=_custom_state_cache_dict[key])
            kw['state'] = state
            return fn(*args, spec=spec, phases=phases, **kw)
        return entry  # type: ignore
    return deco

def default_activation_threshold(spec: Spec) -> int:
    """
    Helper method to use the default balance activation threshold for state creation for tests.
    Usage: `@with_custom_state(threshold_fn=default_activation_threshold, ...)`
    """
    if is_post_electra(spec):
        return spec.MIN_ACTIVATION_BALANCE
    else:
        return spec.MAX_EFFECTIVE_BALANCE

def zero_activation_threshold(spec: Spec) -> int:
    """
    Helper method to use 0 gwei as the activation threshold for state creation for tests.
    Usage: `@with_custom_state(threshold_fn=zero_activation_threshold, ...)`
    """
    return 0

def default_balances(spec: Spec) -> List[int]:
    """
    Helper method to create a series of default balances.
    Usage: `@with_custom_state(balances_fn=default_balances, ...)`
    """
    num_validators = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def default_balances_electra(spec: Spec) -> List[int]:
    """
    Helper method to create a series of default balances for Electra.
    Usage: `@with_custom_state(balances_fn=default_balances_electra, ...)`
    """
    if not is_post_electra(spec):
        return default_balances(spec)
    num_validators = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE_ELECTRA] * num_validators

def scaled_churn_balances_min_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    (This is *firmly* over the churn limit -- thus the +2 instead of just +1)
    See the second argument of ``max`` in ``get_validator_churn_limit``.
    Usage: `@with_custom_state(balances_fn=scaled_churn_balances_min_churn_limit, ...)`
    """
    num_validators = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MIN_PER_EPOCH_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_equal_activation_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    Usage: `@with_custom_state(balances_fn=scaled_churn_balances_equal_activation_churn_limit, ...)`
    """
    num_validators = spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    (This is *firmly* over the churn limit -- thus the +2 instead of just +1)
    Usage: `@with_custom_state(balances_fn=scaled_churn_balances_exceed_activation_churn_limit, ...)`
    """
    num_validators = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_exit_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    (The number of validators is double the amount need for the max activation/exit churn limit)
    Usage: `@with_custom_state(balances_fn=scaled_churn_balances_exceed_activation_churn_limit, ...)`
    """
    num_validators = 2 * spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT // spec.MIN_ACTIVATION_BALANCE
    return [spec.MIN_ACTIVATION_BALANCE] * num_validators

with_state = with_custom_state(default_balances, default_activation_threshold)

def low_balances(spec: Spec) -> List[int]:
    """
    Helper method to create a series of low balances.
    Usage: `@with_custom_state(balances_fn=low_balances, ...)`
    """
    num_validators = spec.SLOTS_PER_EPOCH * 8
    low_balance = 18 * 10 ** 9
    return [low_balance] * num_validators

def misc_balances(spec: Spec) -> List[int]:
    """
    Helper method to create a series of balances that includes some misc. balances.
    Usage: `@with_custom_state(balances_fn=misc_balances, ...)`
    """
    num_validators = spec.SLOTS_PER_EPOCH * 8
    balances = [spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators for i in range(num_validators)]
    rng = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_electra(spec: Spec) -> List[int]:
    """
    Helper method to create a series of balances that includes some misc. balances for Electra.
    Usage: `@with_custom_state(balances_fn=misc_balances, ...)`
    """
    if not is_post_electra(spec):
        return misc_balances(spec)
    num_validators = spec.SLOTS_PER_EPOCH * 8
    balances = [spec.MAX_EFFECTIVE_BALANCE_ELECTRA * 2 * i // num_validators for i in range(num_validators)]
    rng = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_in_default_range_with_many_validators(spec: Spec) -> List[int]:
    """
    Helper method to create a series of balances that includes some misc. balances but
    none that are below the ``EJECTION_BALANCE``.
    """
    num_validators = spec.SLOTS_PER_EPOCH * 8 * 2
    floor = spec.config.EJECTION_BALANCE + spec.EFFECTIVE_BALANCE_INCREMENT
    balances = [max(spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators, floor) for i in range(num_validators)]
    rng = Random(1234)
    rng.shuffle(balances)
    return balances

def low_single_balance(spec: Spec) -> List[int]:
    """
    Helper method to create a single of balance of 1 Gwei.
    Usage: `@with_custom_state(balances_fn=low_single_balance, ...)`
    """
    return [1]

def large_validator_set(spec: Spec) -> List[int]:
    """
    Helper method to create a large series of default balances.
    Usage: `@with_custom_state(balances_fn=default_balances, ...)`
    """
    num_validators = 2 * spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def single_phase(fn: F) -> F:
    """
    Decorator that filters out the phases data.
    most state tests only focus on behavior of a single phase (the "spec").
    This decorator is applied as part of spec_state_test(fn).
    """

    def entry(*args: Any, **kw: Any) -> Any:
        if 'phases' in kw:
            kw.pop('phases')
        return fn(*args, **kw)
    return entry  # type: ignore

DEFAULT_BLS_ACTIVE = True
is_pytest = True

def dump_skipping_message(reason: str) -> None:
    message = f'[Skipped test] {reason}'
    if is_pytest:
        pytest.skip(message)
    else:
        raise SkippedTest(message)

def description(case_description: str) -> Callable[[F], F]:

    def entry(fn: F) -> F:
        return with_meta_tags({'description': case_description})(fn)
    return entry

def spec_test(fn: F) -> F:
    return vector_test()(bls_switch(fn))

def spec_state_test(fn: F) -> F:
    return spec_test(with_state(single_phase(fn)))

def spec_configured_state_test(conf: Dict[str, Any]) -> Callable[[F], F]:
    overrides = with_config_overrides(conf)

    def decorator(fn: F) -> F:
        return spec_test(overrides(with_state(single_phase(fn)))
    return decorator

def _check_current_version(spec: Spec, state: Any, version_name: str) -> bool:
    fork_version_field = version_name.upper() + '_FORK_VERSION'
    try:
        fork_version = getattr(spec.config, fork_version_field)
    except Exception:
        return False
    else:
        return state.fork.current_version == fork_version

def config_fork_epoch_overrides(spec: Spec, state: Any) -> Dict[str, Any]:
    if state.fork.current_version == spec.config.GENESIS_FORK_VERSION:
        return {}
    for fork in ALL_PHASES:
        if fork != PHASE0 and _check_current_version(spec, state, fork):
            overrides = {}
            for f in ALL_PHASES:
                if f != PHASE0 and is_post_fork(fork, f):
                    overrides[f.upper() + '_FORK_EPOCH'] = spec.GENESIS_EPOCH
            return overrides
    return {}

def with_matching_spec_config(emitted_fork: Optional[str] = None) -> Callable[[F], F]:

    def decorator(fn: F) -> F:

        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Any:
            overrides = config_fork_epoch_overrides(spec, kw['state'])
            deco = with_config_overrides(overrides, emitted_fork)
            return deco(fn)(*args, spec=spec, **kw)
        return wrapper
    return decorator

def spec_state_test_with_matching_config(fn: F) -> F:
    return spec_test(with_state(with_matching_spec_config()(single_phase(fn))))

def expect_assertion_error(fn: Callable[[], Any]) -> None:
    bad = False
    try:
        fn()
        bad = True
    except AssertionError:
        pass
    except IndexError:
        pass
    if bad:
        raise AssertionError('expected an assertion error, but got none.')

def never_bls(fn: F) -> F:
    """
    Decorator to apply on ``bls_switch`` decorator to force BLS de-activation. Useful to mark tests as BLS-ignorant.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """

    def entry(*args: Any, **kw: Any) -> Any:
        kw['bls_active'] = False
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 2})(entry)

def always_bls(fn: F) -> F:
    """
    Decorator to apply on ``bls_switch`` decorator to force BLS activation. Useful to mark tests as BLS-dependent.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """

    def entry(*args: Any, **kw: Any) -> Any:
        kw['bls_active'] = True
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 1})(entry)

def bls_switch(fn: F) -> F:
    """
    Decorator to make a function execute with BLS ON, or BLS off.
    Based on an optional bool argument ``bls_active``, passed to the function at runtime.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """

    def entry(*args: Any, **kw: Any) -> Iterator[Any]:
        old_state = bls.bls_active
        bls.bls_active = kw.pop('bls_active', DEFAULT_BLS_ACTIVE)
        res = fn(*args, **kw)
        if res is not None:
            yield from res
        bls.bls_active = old_state
    return entry  # type: ignore

def disable_process_reveal_deadlines(fn: F) -> F:
    """
    Decorator to make a function execute with `process_reveal_deadlines` OFF.
    This is for testing long-range epochs transition without considering the reveal-deadline slashing effect.
    """

    def entry(*args: Any, spec: Spec, **kw: Any) -> Iterator[Any]:
        if hasattr(spec, 'process_reveal_deadlines'):
            old_state = spec.process_reveal_deadlines
            spec.process_reveal_deadlines = lambda state: None
        yield from fn(*args, spec=spec, **kw)
        if hasattr(spec, 'process_reveal_deadlines'):
            spec.process_reveal_deadlines = old_state
    return with_meta_tags({'reveal_deadlines_setting': 1})(entry)

def with_all_phases(fn: F) -> F:
    """
    A decorator for running a test with every phase
    """
    return with_phases(ALL_PHASES)(fn)

def with_all_phases_from(earliest_phase: str, all_phases: List[str] = ALL_PHASES) -> Callable[[F], F]:
    """
    A decorator factory for running a tests with every phase except the ones listed
    """

    def decorator(fn: F) -> F:
        return with_phases([phase for phase in all_phases if is_post_fork(phase, earliest_phase)])(fn)
    return decorator

def with_all_phases_from_except(earliest_phase: str, except_phases: Optional[List[str]] = None) -> Callable[[F], F]:
    """
    A decorator factory for running a tests with every phase except the ones listed
    """
    return with_all_phases_from(earliest_phase, [phase for phase in ALL_PHASES if phase not in except_phases])

def with_all_phases_from_to(from_phase: str, to_phase: str, other_phases: Optional[List[str]] = None, all_phases: List[str] = ALL_PHASES) -> Callable[[F], F]:
    """
    A decorator factory for running a tests with every phase
    from a given start phase up to and excluding a given end phase
    """

    def decorator(fn: F) -> F:
        return with_phases([phase for phase in all_phases if phase !=