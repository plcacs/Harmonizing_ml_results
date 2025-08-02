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
from typing import Any, Callable, Sequence, Dict, List, Optional, Set, Tuple, Generator, TypeVar, Union, Iterator, cast
from lru import LRU

DEFAULT_TEST_PRESET: str = MINIMAL
DEFAULT_PYTEST_FORKS: List[str] = ALL_PHASES

@dataclass(frozen=True)
class ForkMeta:
    pre_fork_name: str
    post_fork_name: str
    fork_epoch: Optional[int] = None

def _prepare_state(balances_fn: Callable[[Spec], List[int]], 
                  threshold_fn: Callable[[Spec], int], 
                  spec: Spec, 
                  phases: Dict[str, Spec]) -> Any:
    balances = balances_fn(spec)
    activation_threshold = threshold_fn(spec)
    state = create_genesis_state(spec=spec, validator_balances=balances, activation_threshold=activation_threshold)
    return state

_custom_state_cache_dict: LRU = LRU(size=10)

def with_custom_state(balances_fn: Callable[[Spec], List[int]], 
                     threshold_fn: Callable[[Spec], int]) -> Callable:
    def deco(fn: Callable) -> Callable:
        def entry(*args: Any, spec: Spec, phases: Dict[str, Spec], **kw: Any) -> Any:
            key = (spec.fork, spec.config.__hash__(), spec.__file__, balances_fn, threshold_fn)
            global _custom_state_cache_dict
            if key not in _custom_state_cache_dict:
                state = _prepare_state(balances_fn, threshold_fn, spec, phases)
                _custom_state_cache_dict[key] = state.get_backing()
            state = spec.BeaconState(backing=_custom_state_cache_dict[key])
            kw['state'] = state
            return fn(*args, spec=spec, phases=phases, **kw)
        return entry
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

def single_phase(fn: Callable) -> Callable:
    """
    Decorator that filters out the phases data.
    most state tests only focus on behavior of a single phase (the "spec").
    This decorator is applied as part of spec_state_test(fn).
    """
    def entry(*args: Any, **kw: Any) -> Any:
        if 'phases' in kw:
            kw.pop('phases')
        return fn(*args, **kw)
    return entry

DEFAULT_BLS_ACTIVE: bool = True
is_pytest: bool = True

def dump_skipping_message(reason: str) -> None:
    message = f'[Skipped test] {reason}'
    if is_pytest:
        pytest.skip(message)
    else:
        raise SkippedTest(message)

def description(case_description: str) -> Callable[[Callable], Callable]:
    def entry(fn: Callable) -> Callable:
        return with_meta_tags({'description': case_description})(fn)
    return entry

def spec_test(fn: Callable) -> Callable:
    return vector_test()(bls_switch(fn))

def spec_state_test(fn: Callable) -> Callable:
    return spec_test(with_state(single_phase(fn)))

def spec_configured_state_test(conf: Dict[str, Any]) -> Callable[[Callable], Callable]:
    overrides = with_config_overrides(conf)
    def decorator(fn: Callable) -> Callable:
        return spec_test(overrides(with_state(single_phase(fn))))
    return decorator

def _check_current_version(spec: Spec, state: Any, version_name: str) -> bool:
    fork_version_field = version_name.upper() + '_FORK_VERSION'
    try:
        fork_version = getattr(spec.config, fork_version_field)
    except Exception:
        return False
    else:
        return state.fork.current_version == fork_version

def config_fork_epoch_overrides(spec: Spec, state: Any) -> Dict[str, int]:
    if state.fork.current_version == spec.config.GENESIS_FORK_VERSION:
        return {}
    for fork in ALL_PHASES:
        if fork != PHASE0 and _check_current_version(spec, state, fork):
            overrides: Dict[str, int] = {}
            for f in ALL_PHASES:
                if f != PHASE0 and is_post_fork(fork, f):
                    overrides[f.upper() + '_FORK_EPOCH'] = spec.GENESIS_EPOCH
            return overrides
    return {}

def with_matching_spec_config(emitted_fork: Optional[str] = None) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Any:
            overrides = config_fork_epoch_overrides(spec, kw['state'])
            deco = with_config_overrides(overrides, emitted_fork)
            return deco(fn)(*args, spec=spec, **kw)
        return wrapper
    return decorator

def spec_state_test_with_matching_config(fn: Callable) -> Callable:
    return spec_test(with_state(with_matching_spec_config()(single_phase(fn))))

def expect_assertion_error(fn: Callable) -> None:
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

def never_bls(fn: Callable) -> Callable:
    """
    Decorator to apply on ``bls_switch`` decorator to force BLS de-activation. Useful to mark tests as BLS-ignorant.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """
    def entry(*args: Any, **kw: Any) -> Any:
        kw['bls_active'] = False
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 2})(entry)

def always_bls(fn: Callable) -> Callable:
    """
    Decorator to apply on ``bls_switch`` decorator to force BLS activation. Useful to mark tests as BLS-dependent.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """
    def entry(*args: Any, **kw: Any) -> Any:
        kw['bls_active'] = True
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 1})(entry)

def bls_switch(fn: Callable) -> Callable:
    """
    Decorator to make a function execute with BLS ON, or BLS off.
    Based on an optional bool argument ``bls_active``, passed to the function at runtime.
    This decorator may only be applied to yielding spec test functions, and should be wrapped by vector_test,
     as the yielding needs to complete before setting back the BLS setting.
    """
    def entry(*args: Any, **kw: Any) -> Optional[Iterator[Any]]:
        old_state = bls.bls_active
        bls.bls_active = kw.pop('bls_active', DEFAULT_BLS_ACTIVE)
        res = fn(*args, **kw)
        if res is not None:
            yield from res
        bls.bls_active = old_state
    return entry

def disable_process_reveal_deadlines(fn: Callable) -> Callable:
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

def with_all_phases(fn: Callable) -> Callable:
    """
    A decorator for running a test with every phase
    """
    return with_phases(ALL_PHASES)(fn)

def with_all_phases_from(earliest_phase: str, all_phases: List[str] = ALL_PHASES) -> Callable[[Callable], Callable]:
    """
    A decorator factory for running a tests with every phase except the ones listed
    """
    def decorator(fn: Callable) -> Callable:
        return with_phases([phase for phase in all_phases if is_post_fork(phase, earliest_phase)])(fn)
    return decorator

def with_all_phases_from_except(earliest_phase: str, except_phases: Optional[List[str]] = None) -> Callable[[Callable], Callable]:
    """
    A decorator factory for running a tests with every phase except the ones listed
    """
    return with_all_phases_from(earliest_phase, [phase for phase in ALL_PHASES if phase not in (except_phases or [])])

def with_all_phases_from_to(from_phase: str, to_phase: str, other_phases: Optional[List[str]] = None, all_phases: List[str] = ALL_PHASES) -> Callable[[Callable], Callable]:
    """
    A decorator factory for running a tests with every phase
    from a given start phase up to and excluding a given end phase
    """
    def decorator(fn: Callable) -> Callable:
        return with_phases([phase for phase in all_phases if phase != to_phase and is_post_fork(to_phase, phase) and is_post_fork(phase, from_phase)], other_phases=other_phases)(fn)
    return decorator

def with_all_phases_except(exclusion_phases: List[str]) -> Callable[[Callable], Callable]:
    """
    A decorator factory for running a tests with every phase except the ones listed
    """
    def decorator(fn: Callable) -> Callable:
        return with_phases([phase for phase in ALL_PHASES if phase not in exclusion_phases])(fn)
    return decorator

def _get_preset_targets(kw: Dict[str, Any]) -> Dict[str, Spec]:
    preset_name = DEFAULT_TEST_PRESET
    if 'preset' in kw:
        preset_name = kw.pop('preset')
    return spec_targets[preset_name]

def _get_run_phases(phases: List[str], kw: Dict[str, Any]) -> Optional[List[str]]:
    """
    Return the fork names for the base `spec` in test cases
    """
    if 'phase' in kw:
        phase = kw.pop('phase')
        if phase not in phases:
            dump_skipping_message(f"doesn't support this fork: {phase}")
            return None
        run_phases = [phase]
    else:
        run_phases = list(set(phases).intersection(DEFAULT_PYTEST_FORKS))
    return run_phases

def _get_available_phases(run_phases: List[str], other_phases: Optional[List[str]]) -> Set[str]:
    """
    Return the available fork names for multi-phase tests
    """
    available_phases = set(run_phases)
    if other_phases is not None:
        available_phases |= set(other_phases)
    return available_phases

def _run_test_case_with_phases(fn: Callable, phases: List[str], other_phases: Optional[List[str]], 
                              kw: Dict[str, Any], args: Tuple[Any, ...], is_fork_transition: bool = False) -> Optional[Any]:
    run_phases = _get_run_phases(phases, kw)
    if run_phases is None or len(run_phases) == 0:
        if not is_fork_transition:
            dump_skipping_message('none of the recognized phases are executable, skipping test.')
        return None
    available_phases = _get_available_phases(run_phases, other_phases)
    targets = _get_preset_targets(kw)
    phase_dir: Dict[str, Spec] = {}
    for phase in available_phases:
        phase_dir[phase] = targets[phase]
    ret = None
    for phase in run_phases:
        ret = fn(*args, spec=targets[phase], phases=phase_dir, **kw)
    return ret

def with_phases(phases: List[str], other_phases: Optional[List[str]] = None) -> Callable[[Callable], Callable]:
    """
    Decorator factory that returns a decorator that runs a test for the appropriate phases.
    Additional phases that do not initially run, but are made available through the test, are optional.
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kw: Any) -> Optional[Any]:
            if 'fork_metas' in kw:
                fork_metas = kw.pop('fork_metas')
                if 'phase' in kw:
                    phase = kw['phase']
                    _phases = [phase]
                    _other_phases = [POST_FORK_OF[phase]]
                    ret = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
                else:
                    ret = None
                    for fork_meta in fork_metas:
                        _phases = [fork_meta.pre_fork_name]
                        _other_phases = [fork_meta.post_fork_name]
                        ret = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
            else:
                ret = _run_test_case_with_phases(fn, phases, other_phases, kw, args)
            return ret
        return wrapper
    return decorator

def with_presets(preset_bases: List[str], reason: Optional[str] = None) -> Callable[[Callable], Callable]:
    available_presets = set(preset_bases)
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Optional[Any]:
            if spec.config.PRESET_BASE not in available_presets:
                message = f"doesn't support this preset base: {spec.config.PRESET_BASE}."
                if reason is not None:
                    message = f'{message} Reason: {reason}'
                dump_skipping_message(message)
                return None
            return fn(*args, spec=spec, **kw)
        return wrapper
    return decorator

with_light_client = with_phases(LIGHT_CLIENT_TESTING_FORKS)
with_altair_and_later = with_all_phases_from(ALTAIR)
with_bellatrix_and_later = with_all_phases_from(BELLATRIX)
with_capella_and_later = with_all_phases_from(CAPELLA)
with_deneb_and_later = with_all_phases_from(DENEB)
with_electra_and_later = with_all_phases_from(ELECTRA)
with_whisk_and_later = with_all_phases_from(WHISK, all_phases=ALLOWED_TEST_RUNNER_FORKS)
with_fulu_and_later = with_all_phases_from(FULU, all_phases=ALLOWED_TEST_RUNNER_FORKS)

class quoted_str(str):
    pass

def _get_basic_dict(ssz_dict: Dict[str, Any]) -> Dict[str, Union[int, bytes, str]]:
    """
    Get dict of basic types from a dict of SSZ objects.
    """
    result: Dict[str, Union[int, bytes, str]] = {}
    for k, v in ssz_dict.items():
        if isinstance(v, int):
            value: Union[int, bytes, str] = int(v)
        elif isinstance(v, bytes):
            value = bytes(bytearray(v))
        else:
            value = quoted_str(v)
        result[k] = value
    return result

def get_copy_of_spec(spec: Spec) -> Spec:
    fork = spec.fork
    preset = spec.config.PRESET_BASE
    module_path = f'eth2spec.{fork}.{preset}'
    module_spec = importlib.util.find_spec(module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.config = deepcopy(spec.config)
    return module

def spec_with_config_overrides(spec: Spec, config_overrides: Dict[str, Any]) -> Tuple[Spec, Dict[str, Any]]:
    config = spec.config._asdict()
    config.update(((k, config_overrides[k]) for k in config.keys() & config_overrides.keys()))
    config_types = spec.Configuration.__annotations__
    modified_config = {k: config_types[k](v) for k, v in config.items()}
    spec.config = spec.Configuration(**modified_config)
    output_config = _get_basic_dict(modified_config)
    return (spec, output_config)

def with_config_overrides(config_overrides: Dict[str, Any], emitted_fork: Optional[str] = None, emit: bool = True) -> Callable[[Callable], Callable]:
    """
    WARNING: the spec_test decorator must wrap this, to ensure the decorated test actually runs.
    This decorator forces the test to yield, and pytest doesn't run generator tests, and instead silently passes it.
    Use 'spec_configured_state_test' instead of 'spec_state_test' if you are unsure.

    This is a decorator that applies a dict of config value overrides to the spec during execution.
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Optional[Iterator[Tuple[str, str, Any]]]:
            spec, output_config = spec_with_config_overrides(get_copy_of_spec(spec), config_overrides)
            if 'phases' in kw:
                phases: Dict[str, Spec] = {}
                for fork in kw['phases']:
                    phases[fork], output = spec_with_config_overrides(get_copy_of_spec(kw['phases'][fork]), config_overrides)
                    if emitted_fork == fork:
                        output_config = output
                kw['phases'] = phases
            if emit:
                yield ('config', 'cfg', output_config)
            out = fn(*args, spec=spec, **kw)
            if out is not None:
                yield from out
        return wrapper
    return decorator

def only_generator(reason: str) -> Callable[[Callable], Callable]:
    def _decorator(inner: Callable) -> Callable:
        def _wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
            if is_pytest:
                dump_skipping_message(reason)
                return None
            return inner(*args, **kwargs)
        return _wrapper
    return _decorator

def with_test_suite_name(suite_name: str) -> Callable[[Callable], Callable]:
    def _decorator(inner: Callable) -> Callable:
        inner.suite_name = suite_name
        return inner
    return _decorator

def set_fork_metas(fork_metas: List[ForkMeta]) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, fork_metas=fork_metas, **kwargs)
        return wrapper
    return decorator

def with_fork_metas(fork_metas: List[ForkMeta]) -> Callable[[Callable], Callable]:
    """
    A decorator to construct a "transition" test from one fork to another.

    Decorator takes a list of `ForkMeta` and each item defines `pre_fork_name`,
    `post_fork_name`, and `fork_epoch`.

    Decorator assumes a transition from the `pre_fork_name` fork to the
    `post_fork_name` fork. The user can supply a `fork_epoch` at which the
    fork occurs or they must compute one (yielding to the generator) during the test
    if more custom behavior is desired.

    A test using this decorator should expect to receive as parameters:
    `state`: the default state constructed for the `pre_fork_name` fork
        according to the `with_state` decorator.
    `fork_epoch`: the `fork_epoch` provided to this decorator, if given.
    `spec`: the version of the eth2 spec corresponding to `pre_fork_name`.
    `post_spec`: the version of the eth2 spec corresponding to `post_fork_name`.
    `pre_tag`: a function to tag data as belonging to `pre_fork_name` fork.
        Used to discriminate data during consumption of the generated spec tests.
    `post_tag`: a function to tag data as belonging to `post_fork_name` fork.
        Used to discriminate data during consumption of the generated spec tests.
    """
    run_yield_fork_meta = yield_fork_meta(fork_metas)
    run_with_phases = with_phases(ALL_PHASES)
    run_set_fork_metas = set_fork_metas(fork_metas)

    def decorator(fn: Callable) -> Callable:
        return run_set_fork_metas(run_with_phases(spec_test(with_state(run_yield_fork_meta(fn)))))
    return decorator

def yield_fork_meta(fork_metas: List[ForkMeta]) -> Callable[[Callable], Callable]:
    """
    Yield meta fields to `meta.yaml` and pass post spec and meta fields to `fn`.
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kw: Any) -> Optional[Iterator[Tuple[str, str, Any]]]:
            phases = kw.pop('phases')
            spec = kw['spec']
            try:
                fork_meta = next(filter(lambda m: m.pre_fork_name == spec.fork, fork_metas))
            except StopIteration:
                dump_skipping_message(f"doesn't support this fork: {spec.fork}")
                return None
            post_spec = phases[fork_meta.post_fork_name]
            pre_fork_counter = 0

            def pre_tag(obj: Any) -> Any:
                nonlocal pre_fork_counter
                pre_fork_counter += 1
                return obj

            def post_tag(obj: Any) -> Any:
                return obj
            
            yield ('post_fork', 'meta', fork_meta.post_fork_name)
            has_fork_epoch = False
            if fork_meta.fork_epoch:
                kw['fork_epoch'] = fork_meta.fork_epoch
                has_fork_epoch = True
                yield ('fork_epoch', 'meta', fork_meta.fork_epoch)
            result = fn(*args, post_spec=post_spec, pre_tag=pre_tag, post_tag=post_tag, **kw)
            if result is not None:
                for part in result:
                    if part[0] == 'fork_epoch':
                        has_fork_epoch = True
                    yield part
            assert has_fork_epoch
            if pre_fork_counter > 0:
                yield ('fork_block', 'meta', pre_fork_counter - 1)
        return wrapper
    return decorator
