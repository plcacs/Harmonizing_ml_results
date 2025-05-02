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
from typing import Any, Callable, Sequence, Dict, TypeVar, Optional, List, Set, Tuple, Union, Generator, Iterator
from lru import LRU
from types import ModuleType

DEFAULT_TEST_PRESET: str = MINIMAL
DEFAULT_PYTEST_FORKS: List[str] = ALL_PHASES

@dataclass(frozen=True)
class ForkMeta:
    pass

T = TypeVar('T')
Fn = TypeVar('Fn', bound=Callable[..., Any])

def _prepare_state(balances_fn: Callable[[Spec], List[int]], threshold_fn: Callable[[Spec], int], spec: Spec, phases: Dict[str, Spec]) -> Any:
    balances: List[int] = balances_fn(spec)
    activation_threshold: int = threshold_fn(spec)
    state: Any = create_genesis_state(spec=spec, validator_balances=balances, activation_threshold=activation_threshold)
    return state

_custom_state_cache_dict: LRU = LRU(size=10)

def with_custom_state(balances_fn: Callable[[Spec], List[int]], threshold_fn: Callable[[Spec], int]) -> Callable[[Fn], Fn]:
    def deco(fn: Fn) -> Fn:
        def entry(*args: Any, spec: Spec, phases: Dict[str, Spec], **kw: Any) -> Any:
            key: Tuple[str, int, str, Callable[[Spec], List[int]], Callable[[Spec], int]] = (spec.fork, spec.config.__hash__(), spec.__file__, balances_fn, threshold_fn)
            global _custom_state_cache_dict
            if key not in _custom_state_cache_dict:
                state: Any = _prepare_state(balances_fn, threshold_fn, spec, phases)
                _custom_state_cache_dict[key] = state.get_backing()
            state: Any = spec.BeaconState(backing=_custom_state_cache_dict[key])
            kw['state'] = state
            return fn(*args, spec=spec, phases=phases, **kw)
        return entry  # type: ignore
    return deco

def default_activation_threshold(spec: Spec) -> int:
    if is_post_electra(spec):
        return spec.MIN_ACTIVATION_BALANCE
    else:
        return spec.MAX_EFFECTIVE_BALANCE

def zero_activation_threshold(spec: Spec) -> int:
    return 0

def default_balances(spec: Spec) -> List[int]:
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def default_balances_electra(spec: Spec) -> List[int]:
    if not is_post_electra(spec):
        return default_balances(spec)
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE_ELECTRA] * num_validators

def scaled_churn_balances_min_churn_limit(spec: Spec) -> List[int]:
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MIN_PER_EPOCH_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_equal_activation_churn_limit(spec: Spec) -> List[int]:
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_churn_limit(spec: Spec) -> List[int]:
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_exit_churn_limit(spec: Spec) -> List[int]:
    num_validators: int = 2 * spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT // spec.MIN_ACTIVATION_BALANCE
    return [spec.MIN_ACTIVATION_BALANCE] * num_validators

with_state: Callable[[Fn], Fn] = with_custom_state(default_balances, default_activation_threshold)

def low_balances(spec: Spec) -> List[int]:
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    low_balance: int = 18 * 10 ** 9
    return [low_balance] * num_validators

def misc_balances(spec: Spec) -> List[int]:
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    balances: List[int] = [spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_electra(spec: Spec) -> List[int]:
    if not is_post_electra(spec):
        return misc_balances(spec)
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    balances: List[int] = [spec.MAX_EFFECTIVE_BALANCE_ELECTRA * 2 * i // num_validators for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_in_default_range_with_many_validators(spec: Spec) -> List[int]:
    num_validators: int = spec.SLOTS_PER_EPOCH * 8 * 2
    floor: int = spec.config.EJECTION_BALANCE + spec.EFFECTIVE_BALANCE_INCREMENT
    balances: List[int] = [max(spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators, floor) for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances

def low_single_balance(spec: Spec) -> List[int]:
    return [1]

def large_validator_set(spec: Spec) -> List[int]:
    num_validators: int = 2 * spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def single_phase(fn: Fn) -> Fn:
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

def description(case_description: str) -> Callable[[Fn], Fn]:
    def entry(fn: Fn) -> Fn:
        return with_meta_tags({'description': case_description})(fn)
    return entry

def spec_test(fn: Fn) -> Fn:
    return vector_test()(bls_switch(fn))

def spec_state_test(fn: Fn) -> Fn:
    return spec_test(with_state(single_phase(fn)))

def spec_configured_state_test(conf: Dict[str, Any]) -> Callable[[Fn], Fn]:
    overrides: Callable[[Fn], Fn] = with_config_overrides(conf)
    def decorator(fn: Fn) -> Fn:
        return spec_test(overrides(with_state(single_phase(fn))))
    return decorator

def _check_current_version(spec: Spec, state: Any, version_name: str) -> bool:
    fork_version_field: str = version_name.upper() + '_FORK_VERSION'
    try:
        fork_version: bytes = getattr(spec.config, fork_version_field)
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

def with_matching_spec_config(emitted_fork: Optional[str] = None) -> Callable[[Fn], Fn]:
    def decorator(fn: Fn) -> Fn:
        def wrapper(*args: Any, spec: Spec, **kw: Any) -> Any:
            overrides: Dict[str, Any] = config_fork_epoch_overrides(spec, kw['state'])
            deco: Callable[[Fn], Fn] = with_config_overrides(overrides, emitted_fork)
            return deco(fn)(*args, spec=spec, **kw)
        return wrapper  # type: ignore
    return decorator

def spec_state_test_with_matching_config(fn: Fn) -> Fn:
    return spec_test(with_state(with_matching_spec_config()(single_phase(fn))))

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

def never_bls(fn: Fn) -> Fn:
    def entry(*args: Any, **kw: Any) -> Any:
        kw['bls_active'] = False
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 2})(entry)  # type: ignore

def always_bls(fn: Fn) -> Fn:
    def entry(*args: Any, **kw: Any) -> Any:
        kw['bls_active'] = True
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 1})(entry)  # type: ignore

def bls_switch(fn: Fn) -> Fn:
    def entry(*args: Any, **kw: Any) -> Generator[Any, None, None]:
        old_state: bool = bls.bls_active
        bls.bls_active = kw.pop('bls_active', DEFAULT_BLS_ACTIVE)
        res: Any = fn(*args, **kw)
        if res is not None:
            yield from res
        bls.bls_active = old_state
    return entry  # type: ignore

def disable_process_reveal_deadlines(fn: Fn) -> Fn:
    def entry(*args: Any, spec: Spec, **kw: Any) -> Generator[Any, None, None]:
        if hasattr(spec, 'process_reveal_deadlines'):
            old_state: Callable[[Any], None] = spec.process_reveal_deadlines
            spec.process_reveal_deadlines = lambda state: None
        yield from fn(*args, spec=spec, **kw)
        if hasattr(spec, 'process_reveal_deadlines'):
            spec.process_reveal_deadlines = old_state
    return with_meta_tags({'reveal_deadlines_setting': 1})(entry)  # type: ignore

def with_all_phases(fn: Fn) -> Fn:
    return with_phases(ALL_PHASES)(fn)

def with_all_phases_from(earliest_phase: str, all_phases: List[str] = ALL_PHASES) -> Callable[[Fn], Fn]:
    def decorator(fn: Fn) -> Fn:
        return with_phases([phase for phase in all_phases if is_post_fork(phase, earliest_phase)])(fn)
    return decorator

def with_all_phases_from_except(earliest_phase: str, except_phases: Optional[List[str]] = None) -> Callable[[Fn], Fn]:
    return with_all_phases_from(earliest_phase, [phase for phase in ALL_PHASES if phase not in except_phases])

def with_all_phases_from_to(from_phase: str, to_phase: str, other_phases: Optional[List[str]] = None, all_phases: List[str] = ALL_PHASES) -> Callable[[Fn], Fn]:
    def decorator(fn: Fn) -> Fn:
        return with_phases([phase for phase in all_phases if phase != to_phase and is_post_fork(to_phase, phase) and is_post_fork(phase, from_phase)], other_phases=other_phases)(fn)
    return decorator

def with_all_phases_except(exclusion_phases: List[str]) -> Callable[[Fn], Fn]:
    def decorator(fn: Fn) -> Fn:
        return with_phases([phase for phase in ALL_PHASES if phase not in exclusion_phases])(fn)
    return decorator

def _get_preset_targets(kw: Dict[str, Any]) -> Dict[str, Spec]:
    preset_name: str = DEFAULT_TEST_PRESET
    if 'preset' in kw:
        preset_name = kw.pop('preset')
    return spec_targets[preset_name]

def _get_run_phases(phases: List[str], kw: Dict[str, Any]) -> Optional[List[str]]:
    if 'phase' in kw:
        phase: str = kw.pop('phase')
        if phase not in phases:
            dump_skipping_message(f"doesn't support this fork: {phase}")
            return None
        run_phases: List[str] = [phase]
    else:
        run_phases: List[str] = list(set(phases).intersection(DEFAULT_PYTEST_FORKS))
    return run_phases

def _get_available_phases(run_phases: List[str], other_phases: Optional[List[str]]) -> Set[str]:
    available_phases: Set[str] = set(run_phases)
    if other_phases is not None:
        available_phases |= set(other_phases)
    return available_phases

def _run_test_case_with_phases(fn: Fn, phases: List[str], other_phases: Optional[List[str]], kw: Dict[str, Any], args: Tuple[Any, ...], is_fork_transition: bool = False) -> Optional[Any]:
    run_phases: Optional[List[str]] = _get_run_phases(phases, kw)
    if run_phases is None or len(run_phases) == 0:
        if not is_fork_transition:
            dump_skipping_message('none of the recognized phases are executable, skipping test.')
        return None
    available_phases: Set[str] = _get_available_phases(run_phases, other_phases)
    targets: Dict[str, Spec] = _get_preset_targets(kw)
    phase_dir: Dict[str, Spec] = {}
    for phase in available_phases:
        phase_dir[phase] = targets[phase]
    for phase in run_phases:
        ret: Any = fn(*args, spec=targets[phase], phases=phase_dir, **kw)
    return ret

def with_phases(phases: List[str], other_phases: Optional[List[str]] = None) -> Callable[[Fn], Fn]:
    def decorator(fn: Fn) -> Fn:
        def wrapper(*args: Any, **kw: Any) -> Optional[Any]:
            if 'fork_metas' in kw:
                fork_metas: List[ForkMeta] = kw.pop('fork_metas')
                if 'phase' in kw:
                    phase: str = kw['phase']
                    _phases: List[str] = [phase]
                    _other_phases: List[str] = [POST_FORK_OF[phase]]
                    ret: Optional[Any] = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
                else:
                    for fork_meta in fork_metas:
                        _phases = [fork_meta.pre_fork_name]
                        _other_phases = [fork_meta.post_fork_name]
                        ret = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
            else:
                ret = _run_test_case_with_phases(fn, phases, other_phases, kw