#!/usr/bin/env python3
from copy import deepcopy
from dataclasses import dataclass
import importlib
from random import Random
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple, TypeVar, Union, Iterator

import pytest
from eth2spec.utils import bls

from .exceptions import SkippedTest
from .helpers.constants import (ALTAIR, ALL_PHASES, ALLOWED_TEST_RUNNER_FORKS, BELLATRIX, CAPELLA, DENEB, ELECTRA, FULU,
                                LIGHT_CLIENT_TESTING_FORKS, MINIMAL, PHASE0, POST_FORK_OF, WHISK)
from .helpers.forks import is_post_fork, is_post_electra
from .helpers.genesis import create_genesis_state
from .helpers.typing import Spec
from .helpers.specs import spec_targets
from .utils import vector_test, with_meta_tags
from lru import LRU

# Type variable for a generic function
F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_TEST_PRESET: str = MINIMAL
DEFAULT_PYTEST_FORKS: Sequence[str] = ALL_PHASES

@dataclass(frozen=True)
class ForkMeta:
    # The actual attributes should be defined here as needed.
    pre_fork_name: str
    post_fork_name: str
    fork_epoch: Optional[int]


def _prepare_state(
    balances_fn: Callable[[Spec], List[int]], 
    threshold_fn: Callable[[Spec], int], 
    spec: Spec, 
    phases: Dict[str, Spec]
) -> Any:
    balances: List[int] = balances_fn(spec)
    activation_threshold: int = threshold_fn(spec)
    state: Any = create_genesis_state(spec=spec, validator_balances=balances, activation_threshold=activation_threshold)
    return state


_custom_state_cache_dict: LRU = LRU(size=10)


def with_custom_state(
    balances_fn: Callable[[Spec], List[int]], 
    threshold_fn: Callable[[Spec], int]
) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        def entry(*args: Any, spec: Spec, phases: Dict[str, Spec], **kw: Any) -> Any:
            key: Tuple[Any, ...] = (spec.fork, spec.config.__hash__(), spec.__file__, balances_fn, threshold_fn)
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
    """
    if is_post_electra(spec):
        return spec.MIN_ACTIVATION_BALANCE
    else:
        return spec.MAX_EFFECTIVE_BALANCE


def zero_activation_threshold(spec: Spec) -> int:
    """
    Helper method to use 0 gwei as the activation threshold for state creation for tests.
    """
    return 0


def default_balances(spec: Spec) -> List[int]:
    """
    Helper method to create a series of default balances.
    """
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators


def default_balances_electra(spec: Spec) -> List[int]:
    """
    Helper method to create a series of default balances for Electra.
    """
    if not is_post_electra(spec):
        return default_balances(spec)
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE_ELECTRA] * num_validators


def scaled_churn_balances_min_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    (This is *firmly* over the churn limit -- thus the +2 instead of just +1)
    """
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MIN_PER_EPOCH_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators


def scaled_churn_balances_equal_activation_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    """
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators


def scaled_churn_balances_exceed_activation_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    (This is *firmly* over the churn limit -- thus the +2 instead of just +1)
    """
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators


def scaled_churn_balances_exceed_activation_exit_churn_limit(spec: Spec) -> List[int]:
    """
    Helper method to create enough validators to scale the churn limit.
    (The number of validators is double the amount need for the max activation/exit churn limit)
    """
    num_validators: int = 2 * spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT // spec.MIN_ACTIVATION_BALANCE
    return [spec.MIN_ACTIVATION_BALANCE] * num_validators


with_state: Callable[[F], F] = with_custom_state(default_balances, default_activation_threshold)


def low_balances(spec: Spec) -> List[int]:
    """
    Helper method to create a series of low balances.
    """
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    low_balance: int = 18 * 10 ** 9
    return [low_balance] * num_validators


def misc_balances(spec: Spec) -> List[int]:
    """
    Helper method to create a series of balances that includes some misc. balances.
    """
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    balances: List[int] = [spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances


def misc_balances_electra(spec: Spec) -> List[int]:
    """
    Helper method to create a series of balances that includes some misc. balances for Electra.
    """
    if not is_post_electra(spec):
        return misc_balances(spec)
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    balances: List[int] = [spec.MAX_EFFECTIVE_BALANCE_ELECTRA * 2 * i // num_validators for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances


def misc_balances_in_default_range_with_many_validators(spec: Spec) -> List[int]:
    """
    Helper method to create a series of balances that includes some misc. balances but
    none that are below the ``EJECTION_BALANCE``.
    """
    num_validators: int = spec.SLOTS_PER_EPOCH * 8 * 2
    floor: int = spec.config.EJECTION_BALANCE + spec.EFFECTIVE_BALANCE_INCREMENT
    balances: List[int] = [max(spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators, floor) for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances


def low_single_balance(spec: Spec) -> List[int]:
    """
    Helper method to create a single of balance of 1 Gwei.
    """
    return [1]


def large_validator_set(spec: Spec) -> List[int]:
    """
    Helper method to create a large series of default balances.
    """
    num_validators: int = 2 * spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators


def single_phase(fn: F) -> F:
    """
    Decorator that filters out the phases data.
    """
    def entry(*args: Any, **kw: Any) -> Any:
        if 'phases' in kw:
            kw.pop('phases')
        return fn(*args, **kw)
    return entry  # type: ignore


DEFAULT_BLS_ACTIVE: bool = True
is_pytest: bool = True


def dump_skipping_message(reason: str) -> None:
    message: str = f'[Skipped test] {reason}'
    if is_pytest:
        pytest.skip(message)
    else:
        raise SkippedTest(message)


def description(case_description: str) -> Callable[[F], F]:
    def entry(fn: F) -> F:
        return with_meta_tags({'description': case_description})(fn)  # type: ignore
    return entry


def spec_test(fn: F) -> F:
    return vector_test()(bls_switch(fn))  # type: ignore


def spec_state_test(fn: F) -> F:
    return spec_test(with_state(single_phase(fn)))  # type: ignore


def spec_configured_state_test(conf: Dict[str, Any]) -> Callable[[F], F]:
    overrides = with_config_overrides(conf)
    def decorator(fn: F) -> F:
        return spec_test(overrides(with_state(single_phase(fn))))  # type: ignore
    return decorator


def _check_current_version(spec: Spec, state: Any, version_name: str) -> bool:
    fork_version_field: str = version_name.upper() + '_FORK_VERSION'
    try:
        fork_version: Any = getattr(spec.config, fork_version_field)
    except Exception:
        return False
    else:
        return state.fork.current_version == fork_version


def config_fork_epoch_overrides(spec: Spec, state: Any) -> Dict[str, Any]:
    if state.fork.current_version == spec.config.GENESIS_FORK_VERSION:
        return {}
    for fork in ALL_PHASES:
        if fork != PHASE0 and _check_current_version(spec, state, fork):
            overrides: Dict[str, Any] = {}
            for f in ALL_PHASES:
                if f != PHASE0 and is_post_fork(fork, f):
                    overrides[f.upper() + '_FORK_EPOCH'] = spec.GENESIS_EPOCH
            return overrides
    return {}


def with_matching_spec_config(emitted_fork: Optional[str] = None) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Any:
            overrides: Dict[str, Any] = config_fork_epoch_overrides(spec, kw['state'])
            deco = with_config_overrides(overrides, emitted_fork)
            return deco(fn)(*args, spec=spec, **kw)
        return wrapper  # type: ignore
    return decorator


def spec_state_test_with_matching_config(fn: F) -> F:
    return spec_test(with_state(with_matching_spec_config()(single_phase(fn))))  # type: ignore


def expect_assertion_error(fn: Callable[[], Any]) -> None:
    bad: bool = False
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
    Decorator to force BLS de-activation.
    """
    def entry(*args: Any, **kw: Any) -> Generator[Any, None, None]:
        kw['bls_active'] = False
        gen = bls_switch(fn)(*args, **kw)
        yield from gen
    return with_meta_tags({'bls_setting': 2})(entry)  # type: ignore


def always_bls(fn: F) -> F:
    """
    Decorator to force BLS activation.
    """
    def entry(*args: Any, **kw: Any) -> Generator[Any, None, None]:
        kw['bls_active'] = True
        gen = bls_switch(fn)(*args, **kw)
        yield from gen
    return with_meta_tags({'bls_setting': 1})(entry)  # type: ignore


def bls_switch(fn: F) -> Callable[..., Generator[Any, None, None]]:
    """
    Decorator to make a function execute with BLS ON, or BLS off.
    """
    def entry(*args: Any, **kw: Any) -> Generator[Any, None, None]:
        old_state: bool = bls.bls_active
        bls.bls_active = kw.pop('bls_active', DEFAULT_BLS_ACTIVE)
        res: Any = fn(*args, **kw)
        if res is not None:
            yield from res
        bls.bls_active = old_state
    return entry


def disable_process_reveal_deadlines(fn: F) -> Callable[..., Generator[Any, None, None]]:
    """
    Decorator to execute a function with `process_reveal_deadlines` OFF.
    """
    def entry(*args: Any, spec: Spec, **kw: Any) -> Generator[Any, None, None]:
        if hasattr(spec, 'process_reveal_deadlines'):
            old_state: Any = spec.process_reveal_deadlines
            spec.process_reveal_deadlines = lambda state: None
        gen: Generator[Any, None, None] = fn(*args, spec=spec, **kw)
        yield from gen
        if hasattr(spec, 'process_reveal_deadlines'):
            spec.process_reveal_deadlines = old_state
    return with_meta_tags({'reveal_deadlines_setting': 1})(entry)  # type: ignore


def with_all_phases(fn: F) -> F:
    """
    A decorator for running a test with every phase.
    """
    return with_phases(ALL_PHASES)(fn)  # type: ignore


def with_all_phases_from(earliest_phase: str, all_phases: Sequence[str] = ALL_PHASES) -> Callable[[F], F]:
    """
    A decorator factory for running a test with every phase except those before the earliest_phase.
    """
    def decorator(fn: F) -> F:
        return with_phases([phase for phase in all_phases if is_post_fork(phase, earliest_phase)])(fn)  # type: ignore
    return decorator


def with_all_phases_from_except(earliest_phase: str, except_phases: Optional[Sequence[str]] = None) -> Callable[[F], F]:
    """
    A decorator factory for running a test with every phase except the ones listed.
    """
    except_phases = except_phases or []
    return with_all_phases_from(earliest_phase, [phase for phase in ALL_PHASES if phase not in except_phases])


def with_all_phases_from_to(from_phase: str, to_phase: str, other_phases: Optional[Sequence[str]] = None, all_phases: Sequence[str] = ALL_PHASES) -> Callable[[F], F]:
    """
    A decorator factory for running a test with every phase from a given start phase up to and excluding a given end phase.
    """
    def decorator(fn: F) -> F:
        phases: List[str] = [phase for phase in all_phases if phase != to_phase and is_post_fork(to_phase, phase) and is_post_fork(phase, from_phase)]
        return with_phases(phases, other_phases=other_phases)(fn)  # type: ignore
    return decorator


def with_all_phases_except(exclusion_phases: Sequence[str]) -> Callable[[F], F]:
    """
    A decorator factory for running a test with every phase except the ones listed.
    """
    def decorator(fn: F) -> F:
        phases: List[str] = [phase for phase in ALL_PHASES if phase not in exclusion_phases]
        return with_phases(phases)(fn)  # type: ignore
    return decorator


def _get_preset_targets(kw: Dict[str, Any]) -> Dict[str, Spec]:
    preset_name: str = DEFAULT_TEST_PRESET
    if 'preset' in kw:
        preset_name = kw.pop('preset')
    return spec_targets[preset_name]


def _get_run_phases(phases: Sequence[str], kw: Dict[str, Any]) -> Optional[Set[str]]:
    """
    Return the fork names for the base `spec` in test cases.
    """
    if 'phase' in kw:
        phase: str = kw.pop('phase')
        if phase not in phases:
            dump_skipping_message(f"doesn't support this fork: {phase}")
            return None
        run_phases: Set[str] = {phase}
    else:
        run_phases = set(phases).intersection(DEFAULT_PYTEST_FORKS)
    return run_phases


def _get_available_phases(run_phases: Set[str], other_phases: Optional[Sequence[str]]) -> Set[str]:
    """
    Return the available fork names for multi-phase tests.
    """
    available_phases: Set[str] = set(run_phases)
    if other_phases is not None:
        available_phases |= set(other_phases)
    return available_phases


def _run_test_case_with_phases(
    fn: Callable[..., Any],
    phases: Sequence[str],
    other_phases: Optional[Sequence[str]],
    kw: Dict[str, Any],
    args: Tuple[Any, ...],
    is_fork_transition: bool = False
) -> Any:
    run_phases: Optional[Set[str]] = _get_run_phases(phases, kw)
    if run_phases is None or len(run_phases) == 0:
        if not is_fork_transition:
            dump_skipping_message('none of the recognized phases are executable, skipping test.')
        return None
    available_phases: Set[str] = _get_available_phases(run_phases, other_phases)
    targets: Dict[str, Spec] = _get_preset_targets(kw)
    phase_dir: Dict[str, Spec] = {}
    for phase in available_phases:
        phase_dir[phase] = targets[phase]
    ret: Any = None
    for phase in run_phases:
        ret = fn(*args, spec=targets[phase], phases=phase_dir, **kw)
    return ret


def with_phases(phases: Sequence[str], other_phases: Optional[Sequence[str]] = None) -> Callable[[F], F]:
    """
    Decorator factory that returns a decorator which runs a test for the appropriate phases.
    """
    def decorator(fn: F) -> F:
        def wrapper(*args: Any, **kw: Any) -> Any:
            if 'fork_metas' in kw:
                fork_metas: Sequence[ForkMeta] = kw.pop('fork_metas')
                if 'phase' in kw:
                    phase: str = kw['phase']
                    _phases: List[str] = [phase]
                    _other_phases: List[str] = [POST_FORK_OF[phase]]
                    ret = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
                else:
                    for fork_meta in fork_metas:
                        _phases = [fork_meta.pre_fork_name]
                        _other_phases = [fork_meta.post_fork_name]
                        ret = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
                return ret
            else:
                ret = _run_test_case_with_phases(fn, phases, other_phases, kw, args)
                return ret
        return wrapper  # type: ignore
    return decorator


def with_presets(preset_bases: Sequence[str], reason: Optional[str] = None) -> Callable[[F], F]:
    available_presets: Set[str] = set(preset_bases)
    def decorator(fn: F) -> F:
        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Any:
            if spec.config.PRESET_BASE not in available_presets:
                message: str = f"doesn't support this preset base: {spec.config.PRESET_BASE}."
                if reason is not None:
                    message = f'{message} Reason: {reason}'
                dump_skipping_message(message)
                return None
            return fn(*args, spec=spec, **kw)
        return wrapper  # type: ignore
    return decorator


with_light_client: Callable[[F], F] = with_phases(LIGHT_CLIENT_TESTING_FORKS)
with_altair_and_later: Callable[[F], F] = with_all_phases_from(ALTAIR)
with_bellatrix_and_later: Callable[[F], F] = with_all_phases_from(BELLATRIX)
with_capella_and_later: Callable[[F], F] = with_all_phases_from(CAPELLA)
with_deneb_and_later: Callable[[F], F] = with_all_phases_from(DENEB)
with_electra_and_later: Callable[[F], F] = with_all_phases_from(ELECTRA)
with_whisk_and_later: Callable[[F], F] = with_all_phases_from(WHISK, all_phases=ALLOWED_TEST_RUNNER_FORKS)
with_fulu_and_later: Callable[[F], F] = with_all_phases_from(FULU, all_phases=ALLOWED_TEST_RUNNER_FORKS)


class quoted_str(str):
    pass


def _get_basic_dict(ssz_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Get dict of basic types from a dict of SSZ objects.
    """
    result: Dict[Any, Any] = {}
    for k, v in ssz_dict.items():
        if isinstance(v, int):
            value = int(v)
        elif isinstance(v, bytes):
            value = bytes(bytearray(v))
        else:
            value = quoted_str(v)
        result[k] = value
    return result


def get_copy_of_spec(spec: Spec) -> Spec:
    fork: str = spec.fork
    preset: str = spec.config.PRESET_BASE
    module_path: str = f'eth2spec.{fork}.{preset}'
    module_spec = importlib.util.find_spec(module_path)
    if module_spec is None:
        raise ImportError(f"Cannot find module for {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)  # type: ignore
    module.config = deepcopy(spec.config)
    return module  # type: ignore


def spec_with_config_overrides(spec: Spec, config_overrides: Dict[str, Any]) -> Tuple[Spec, Dict[Any, Any]]:
    config: Dict[str, Any] = spec.config._asdict()
    config.update(((k, config_overrides[k]) for k in config.keys() & config_overrides.keys()))
    config_types: Dict[str, Any] = spec.Configuration.__annotations__
    modified_config: Dict[str, Any] = {k: config_types[k](v) for k, v in config.items()}
    spec.config = spec.Configuration(**modified_config)
    output_config: Dict[Any, Any] = _get_basic_dict(modified_config)
    return spec, output_config


def with_config_overrides(
    config_overrides: Dict[str, Any], 
    emitted_fork: Optional[str] = None, 
    emit: bool = True
) -> Callable[[F], Callable[..., Generator[Any, None, None]]]:
    """
    This decorator applies a dict of config value overrides to the spec during execution.
    """
    def decorator(fn: F) -> Callable[..., Generator[Any, None, None]]:
        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Generator[Any, None, None]:
            spec_copy: Spec = get_copy_of_spec(spec)
            spec_copy, output_config = spec_with_config_overrides(spec_copy, config_overrides)
            if 'phases' in kw:
                phases: Dict[str, Spec] = {}
                for fork in kw['phases']:
                    ph, output = spec_with_config_overrides(get_copy_of_spec(kw['phases'][fork]), config_overrides)
                    if emitted_fork == fork:
                        output_config = output
                    phases[fork] = ph
                kw['phases'] = phases
            if emit:
                yield ('config', 'cfg', output_config)
            out = fn(*args, spec=spec_copy, **kw)
            if out is not None:
                yield from out
        return wrapper
    return decorator


def only_generator(reason: str) -> Callable[[F], F]:
    def _decorator(inner: F) -> F:
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_pytest:
                dump_skipping_message(reason)
                return None
            return inner(*args, **kwargs)
        return _wrapper  # type: ignore
    return _decorator


def with_test_suite_name(suite_name: str) -> Callable[[F], F]:
    def _decorator(inner: F) -> F:
        inner.suite_name = suite_name
        return inner
    return _decorator


def set_fork_metas(fork_metas: Sequence[ForkMeta]) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, fork_metas=fork_metas, **kwargs)
        return wrapper  # type: ignore
    return decorator


def with_fork_metas(fork_metas: Sequence[ForkMeta]) -> Callable[[F], F]:
    """
    A decorator to construct a "transition" test from one fork to another.
    """
    run_yield_fork_meta = yield_fork_meta(fork_metas)
    run_with_phases = with_phases(ALL_PHASES)
    run_set_fork_metas = set_fork_metas(fork_metas)
    def decorator(fn: F) -> F:
        return run_set_fork_metas(run_with_phases(spec_test(with_state(run_yield_fork_meta(fn)))))  # type: ignore
    return decorator


def yield_fork_meta(fork_metas: Sequence[ForkMeta]) -> Callable[[F], Callable[..., Generator[Any, None, None]]]:
    """
    Yield meta fields and pass post spec and meta fields to `fn`.
    """
    def decorator(fn: F) -> Callable[..., Generator[Any, None, None]]:
        def wrapper(*args: Any, **kw: Any) -> Generator[Any, None, None]:
            phases: Dict[str, Spec] = kw.pop('phases')
            spec: Spec = kw['spec']
            try:
                fork_meta: ForkMeta = next(filter(lambda m: m.pre_fork_name == spec.fork, fork_metas))
            except StopIteration:
                dump_skipping_message(f"doesn't support this fork: {spec.fork}")
                return
            post_spec: Spec = phases[fork_meta.post_fork_name]
            pre_fork_counter: int = 0
            def pre_tag(obj: Any) -> Any:
                nonlocal pre_fork_counter
                pre_fork_counter += 1
                return obj
            def post_tag(obj: Any) -> Any:
                return obj
            yield ('post_fork', 'meta', fork_meta.post_fork_name)
            has_fork_epoch: bool = False
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
