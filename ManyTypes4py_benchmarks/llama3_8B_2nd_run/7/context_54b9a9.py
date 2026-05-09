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
from typing import Any, Callable, Sequence, Dict, LRU
from lru import LRU

@dataclass(frozen=True)
class ForkMeta:
    pass

def _prepare_state(balances_fn: Callable[[Spec], Sequence[int]], threshold_fn: Callable[[Spec], int], spec: Spec, phases: Sequence[str]) -> Any:
    balances = balances_fn(spec)
    activation_threshold = threshold_fn(spec)
    state = create_genesis_state(spec=spec, validator_balances=balances, activation_threshold=activation_threshold)
    return state

_custom_state_cache_dict: LRU = LRU(size=10)

def with_custom_state(balances_fn: Callable[[Spec], Sequence[int]], threshold_fn: Callable[[Spec], int]) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        def entry(*args, spec: Spec, phases: Sequence[str], **kw: Dict[str, Any]) -> Any:
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
    if is_post_electra(spec):
        return spec.MIN_ACTIVATION_BALANCE
    else:
        return spec.MAX_EFFECTIVE_BALANCE

def zero_activation_threshold(spec: Spec) -> int:
    return 0

def default_balances(spec: Spec) -> Sequence[int]:
    num_validators = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def default_balances_electra(spec: Spec) -> Sequence[int]:
    if not is_post_electra(spec):
        return default_balances(spec)
    num_validators = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE_ELECTRA] * num_validators

def scaled_churn_balances_min_churn_limit(spec: Spec) -> Sequence[int]:
    num_validators = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MIN_PER_EPOCH_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_equal_activation_churn_limit(spec: Spec) -> Sequence[int]:
    num_validators = spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_churn_limit(spec: Spec) -> Sequence[int]:
    num_validators = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_exit_churn_limit(spec: Spec) -> Sequence[int]:
    num_validators = 2 * spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT // spec.MIN_ACTIVATION_BALANCE
    return [spec.MIN_ACTIVATION_BALANCE] * num_validators

with_state = with_custom_state(default_balances, default_activation_threshold)

def low_balances(spec: Spec) -> Sequence[int]:
    num_validators = spec.SLOTS_PER_EPOCH * 8
    low_balance = 18 * 10 ** 9
    return [low_balance] * num_validators

def misc_balances(spec: Spec) -> Sequence[int]:
    num_validators = spec.SLOTS_PER_EPOCH * 8
    balances = [spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators for i in range(num_validators)]
    rng = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_electra(spec: Spec) -> Sequence[int]:
    if not is_post_electra(spec):
        return misc_balances(spec)
    num_validators = spec.SLOTS_PER_EPOCH * 8
    balances = [spec.MAX_EFFECTIVE_BALANCE_ELECTRA * 2 * i // num_validators for i in range(num_validators)]
    rng = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_in_default_range_with_many_validators(spec: Spec) -> Sequence[int]:
    num_validators = spec.SLOTS_PER_EPOCH * 8 * 2
    floor = spec.config.EJECTION_BALANCE + spec.EFFECTIVE_BALANCE_INCREMENT
    balances = [max(spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators, floor) for i in range(num_validators)]
    rng = Random(1234)
    rng.shuffle(balances)
    return balances

def low_single_balance(spec: Spec) -> Sequence[int]:
    return [1]

def large_validator_set(spec: Spec) -> Sequence[int]:
    num_validators = 2 * spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def single_phase(fn: Callable) -> Callable:
    def entry(*args, **kw: Dict[str, Any]) -> Any:
        if 'phases' in kw:
            kw.pop('phases')
        return fn(*args, **kw)
    return entry

def spec_test(fn: Callable) -> Callable:
    return vector_test()(bls_switch(fn))

def spec_state_test(fn: Callable) -> Callable:
    return spec_test(with_state(single_phase(fn)))

def spec_configured_state_test(conf: Spec) -> Callable:
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

def with_matching_spec_config(emitted_fork: str = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, spec: Spec, **kw: Dict[str, Any]) -> Any:
            overrides = config_fork_epoch_overrides(spec, kw['state'])
            deco = with_config_overrides(overrides, emitted_fork)
            return deco(fn)(*args, spec=spec, **kw)
        return wrapper
    return decorator

def spec_state_test_with_matching_config(fn: Callable) -> Callable:
    return spec_test(with_matching_spec_config()(single_phase(fn)))

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
    def entry(*args, **kw: Dict[str, Any]) -> Any:
        kw['bls_active'] = False
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 2})(entry)

def always_bls(fn: Callable) -> Callable:
    def entry(*args, **kw: Dict[str, Any]) -> Any:
        kw['bls_active'] = True
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 1})(entry)

def bls_switch(fn: Callable) -> Callable:
    def entry(*args, **kw: Dict[str, Any]) -> Any:
        old_state = bls.bls_active
        bls.bls_active = kw.pop('bls_active', DEFAULT_BLS_ACTIVE)
        res = fn(*args, **kw)
        if res is not None:
            yield from res
        bls.bls_active = old_state
    return entry

def disable_process_reveal_deadlines(fn: Callable) -> Callable:
    def entry(*args, spec: Spec, **kw: Dict[str, Any]) -> Any:
        if hasattr(spec, 'process_reveal_deadlines'):
            old_state = spec.process_reveal_deadlines
            spec.process_reveal_deadlines = lambda state: None
        yield from fn(*args, spec=spec, **kw)
        if hasattr(spec, 'process_reveal_deadlines'):
            spec.process_reveal_deadlines = old_state
    return with_meta_tags({'reveal_deadlines_setting': 1})(entry)

def with_all_phases(fn: Callable) -> Callable:
    return with_phases(ALL_PHASES)(fn)

def with_all_phases_from(earliest_phase: str, all_phases: Sequence[str] = ALL_PHASES) -> Callable:
    def decorator(fn: Callable) -> Callable:
        return with_phases([phase for phase in all_phases if is_post_fork(phase, earliest_phase)])(fn)
    return decorator

def with_all_phases_from_except(earliest_phase: str, except_phases: Sequence[str] = []) -> Callable:
    return with_all_phases_from(earliest_phase, [phase for phase in ALL_PHASES if phase not in except_phases])

def with_all_phases_from_to(from_phase: str, to_phase: str, other_phases: Sequence[str] = ALL_PHASES) -> Callable:
    def decorator(fn: Callable) -> Callable:
        return with_phases([phase for phase in other_phases if phase != to_phase and is_post_fork(to_phase, phase) and is_post_fork(phase, from_phase)])(fn)
    return decorator

def with_all_phases_except(exclusion_phases: Sequence[str]) -> Callable:
    def decorator(fn: Callable) -> Callable:
        return with_phases([phase for phase in ALL_PHASES if phase not in exclusion_phases])(fn)
    return decorator

def with_presets(preset_bases: Sequence[str]) -> Callable:
    available_presets = set(preset_bases)

    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, spec: Spec, **kw: