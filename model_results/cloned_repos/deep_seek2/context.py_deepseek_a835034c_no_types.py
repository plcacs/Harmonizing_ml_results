import pytest
from copy import deepcopy
from dataclasses import dataclass
import importlib
from typing import Any, Callable, Sequence, Dict, List, Optional, Set, TypeVar, Generator, Union
from random import Random
from lru import LRU
from eth2spec.utils import bls
from .exceptions import SkippedTest
from .helpers.constants import PHASE0, ALTAIR, BELLATRIX, CAPELLA, DENEB, ELECTRA, FULU, WHISK, MINIMAL, ALL_PHASES, POST_FORK_OF, ALLOWED_TEST_RUNNER_FORKS, LIGHT_CLIENT_TESTING_FORKS
from .helpers.forks import is_post_fork, is_post_electra
from .helpers.genesis import create_genesis_state
from .helpers.typing import Spec, SpecForks
from .helpers.specs import spec_targets
from .utils import vector_test, with_meta_tags
DEFAULT_TEST_PRESET: str = MINIMAL
DEFAULT_PYTEST_FORKS: Set[str] = ALL_PHASES

@dataclass(frozen=True)
class ForkMeta:
    pre_fork_name: str
    post_fork_name: str
    fork_epoch: int

def _prepare_state(balances_fn, threshold_fn, spec, phases):
    balances: Sequence[int] = balances_fn(spec)
    activation_threshold: int = threshold_fn(spec)
    state: Any = create_genesis_state(spec=spec, validator_balances=balances, activation_threshold=activation_threshold)
    return state
_custom_state_cache_dict: LRU = LRU(size=10)

def with_custom_state(balances_fn, threshold_fn):

    def deco(fn):

        def entry(*args: Any, spec: Spec, phases: SpecForks, **kw: Any):
            key: Any = (spec.fork, spec.config.__hash__(), spec.__file__, balances_fn, threshold_fn)
            global _custom_state_cache_dict
            if key not in _custom_state_cache_dict:
                state: Any = _prepare_state(balances_fn, threshold_fn, spec, phases)
                _custom_state_cache_dict[key] = state.get_backing()
            state: Any = spec.BeaconState(backing=_custom_state_cache_dict[key])
            kw['state'] = state
            return fn(*args, spec=spec, phases=phases, **kw)
        return entry
    return deco

def default_activation_threshold(spec):
    if is_post_electra(spec):
        return spec.MIN_ACTIVATION_BALANCE
    else:
        return spec.MAX_EFFECTIVE_BALANCE

def zero_activation_threshold(spec):
    return 0

def default_balances(spec):
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def default_balances_electra(spec):
    if not is_post_electra(spec):
        return default_balances(spec)
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    return [spec.MAX_EFFECTIVE_BALANCE_ELECTRA] * num_validators

def scaled_churn_balances_min_churn_limit(spec):
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MIN_PER_EPOCH_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_equal_activation_churn_limit(spec):
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_churn_limit(spec):
    num_validators: int = spec.config.CHURN_LIMIT_QUOTIENT * (spec.config.MAX_PER_EPOCH_ACTIVATION_CHURN_LIMIT + 2)
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def scaled_churn_balances_exceed_activation_exit_churn_limit(spec):
    num_validators: int = 2 * spec.config.CHURN_LIMIT_QUOTIENT * spec.config.MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT // spec.MIN_ACTIVATION_BALANCE
    return [spec.MIN_ACTIVATION_BALANCE] * num_validators
with_state: Callable = with_custom_state(default_balances, default_activation_threshold)

def low_balances(spec):
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    low_balance: int = 18 * 10 ** 9
    return [low_balance] * num_validators

def misc_balances(spec):
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    balances: List[int] = [spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_electra(spec):
    if not is_post_electra(spec):
        return misc_balances(spec)
    num_validators: int = spec.SLOTS_PER_EPOCH * 8
    balances: List[int] = [spec.MAX_EFFECTIVE_BALANCE_ELECTRA * 2 * i // num_validators for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances

def misc_balances_in_default_range_with_many_validators(spec):
    num_validators: int = spec.SLOTS_PER_EPOCH * 8 * 2
    floor: int = spec.config.EJECTION_BALANCE + spec.EFFECTIVE_BALANCE_INCREMENT
    balances: List[int] = [max(spec.MAX_EFFECTIVE_BALANCE * 2 * i // num_validators, floor) for i in range(num_validators)]
    rng: Random = Random(1234)
    rng.shuffle(balances)
    return balances

def low_single_balance(spec):
    return [1]

def large_validator_set(spec):
    num_validators: int = 2 * spec.SLOTS_PER_EPOCH * spec.MAX_COMMITTEES_PER_SLOT * spec.TARGET_COMMITTEE_SIZE
    return [spec.MAX_EFFECTIVE_BALANCE] * num_validators

def single_phase(fn):

    def entry(*args: Any, **kw: Any):
        if 'phases' in kw:
            kw.pop('phases')
        return fn(*args, **kw)
    return entry
DEFAULT_BLS_ACTIVE: bool = True
is_pytest: bool = True

def dump_skipping_message(reason):
    message: str = f'[Skipped test] {reason}'
    if is_pytest:
        pytest.skip(message)
    else:
        raise SkippedTest(message)

def description(case_description):

    def entry(fn):
        return with_meta_tags({'description': case_description})(fn)
    return entry

def spec_test(fn):
    return vector_test()(bls_switch(fn))

def spec_state_test(fn):
    return spec_test(with_state(single_phase(fn)))

def spec_configured_state_test(conf):
    overrides: Callable = with_config_overrides(conf)

    def decorator(fn):
        return spec_test(overrides(with_state(single_phase(fn))))
    return decorator

def _check_current_version(spec, state, version_name):
    fork_version_field: str = version_name.upper() + '_FORK_VERSION'
    try:
        fork_version: Any = getattr(spec.config, fork_version_field)
    except Exception:
        return False
    else:
        return state.fork.current_version == fork_version

def config_fork_epoch_overrides(spec, state):
    if state.fork.current_version == spec.config.GENESIS_FORK_VERSION:
        return {}
    overrides: Dict[str, Any] = {}
    for fork in ALL_PHASES:
        if fork != PHASE0 and _check_current_version(spec, state, fork):
            for f in ALL_PHASES:
                if f != PHASE0 and is_post_fork(fork, f):
                    overrides[f.upper() + '_FORK_EPOCH'] = spec.GENESIS_EPOCH
            return overrides
    return overrides

def with_matching_spec_config(emitted_fork=None):

    def decorator(fn):

        def wrapper(*args: Any, spec: Spec, **kw: Any):
            overrides: Dict[str, Any] = config_fork_epoch_overrides(spec, kw['state'])
            deco: Callable = with_config_overrides(overrides, emitted_fork)
            return deco(fn)(*args, spec=spec, **kw)
        return wrapper
    return decorator

def spec_state_test_with_matching_config(fn):
    return spec_test(with_state(with_matching_spec_config()(single_phase(fn))))

def expect_assertion_error(fn):
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

def never_bls(fn):

    def entry(*args: Any, **kw: Any):
        kw['bls_active'] = False
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 2})(entry)

def always_bls(fn):

    def entry(*args: Any, **kw: Any):
        kw['bls_active'] = True
        return bls_switch(fn)(*args, **kw)
    return with_meta_tags({'bls_setting': 1})(entry)

def bls_switch(fn):

    def entry(*args: Any, **kw: Any):
        old_state: bool = bls.bls_active
        bls.bls_active = kw.pop('bls_active', DEFAULT_BLS_ACTIVE)
        res: Any = fn(*args, **kw)
        if res is not None:
            yield from res
        bls.bls_active = old_state
    return entry

def disable_process_reveal_deadlines(fn):

    def entry(*args: Any, spec: Spec, **kw: Any):
        if hasattr(spec, 'process_reveal_deadlines'):
            old_state: Any = spec.process_reveal_deadlines
            spec.process_reveal_deadlines = lambda state: None
        yield from fn(*args, spec=spec, **kw)
        if hasattr(spec, 'process_reveal_deadlines'):
            spec.process_reveal_deadlines = old_state
    return with_meta_tags({'reveal_deadlines_setting': 1})(entry)

def with_all_phases(fn):
    return with_phases(ALL_PHASES)(fn)

def with_all_phases_from(earliest_phase, all_phases=ALL_PHASES):

    def decorator(fn):
        return with_phases([phase for phase in all_phases if is_post_fork(phase, earliest_phase)])(fn)
    return decorator

def with_all_phases_from_except(earliest_phase, except_phases=None):
    return with_all_phases_from(earliest_phase, [phase for phase in ALL_PHASES if phase not in except_phases])

def with_all_phases_from_to(from_phase, to_phase, other_phases=None, all_phases=ALL_PHASES):

    def decorator(fn):
        return with_phases([phase for phase in all_phases if phase != to_phase and is_post_fork(to_phase, phase) and is_post_fork(phase, from_phase)], other_phases=other_phases)(fn)
    return decorator

def with_all_phases_except(exclusion_phases):

    def decorator(fn):
        return with_phases([phase for phase in ALL_PHASES if phase not in exclusion_phases])(fn)
    return decorator

def _get_preset_targets(kw):
    preset_name: str = DEFAULT_TEST_PRESET
    if 'preset' in kw:
        preset_name = kw.pop('preset')
    return spec_targets[preset_name]

def _get_run_phases(phases, kw):
    if 'phase' in kw:
        phase: str = kw.pop('phase')
        if phase not in phases:
            dump_skipping_message(f"doesn't support this fork: {phase}")
            return None
        run_phases: List[str] = [phase]
    else:
        run_phases: List[str] = list(set(phases).intersection(DEFAULT_PYTEST_FORKS))
    return run_phases

def _get_available_phases(run_phases, other_phases):
    available_phases: Set[str] = set(run_phases)
    if other_phases is not None:
        available_phases |= set(other_phases)
    return available_phases

def _run_test_case_with_phases(fn, phases, other_phases, kw, args, is_fork_transition=False):
    run_phases: Optional[List[str]] = _get_run_phases(phases, kw)
    if run_phases is None or len(run_phases) == 0:
        if not is_fork_transition:
            dump_skipping_message('none of the recognized phases are executable, skipping test.')
        return None
    available_phases: Set[str] = _get_available_phases(run_phases, other_phases)
    targets: Dict[str, Any] = _get_preset_targets(kw)
    phase_dir: Dict[str, Any] = {}
    for phase in available_phases:
        phase_dir[phase] = targets[phase]
    ret: Optional[Any] = None
    for phase in run_phases:
        ret = fn(*args, spec=targets[phase], phases=phase_dir, **kw)
    return ret

def with_phases(phases, other_phases=None):

    def decorator(fn):

        def wrapper(*args: Any, **kw: Any):
            if 'fork_metas' in kw:
                fork_metas: Sequence[ForkMeta] = kw.pop('fork_metas')
                if 'phase' in kw:
                    phase: str = kw['phase']
                    _phases: List[str] = [phase]
                    _other_phases: List[str] = [POST_FORK_OF[phase]]
                    ret: Optional[Any] = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
                else:
                    ret: Optional[Any] = None
                    for fork_meta in fork_metas:
                        _phases: List[str] = [fork_meta.pre_fork_name]
                        _other_phases: List[str] = [fork_meta.post_fork_name]
                        ret = _run_test_case_with_phases(fn, _phases, _other_phases, kw, args, is_fork_transition=True)
            else:
                ret: Optional[Any] = _run_test_case_with_phases(fn, phases, other_phases, kw, args)
            return ret
        return wrapper
    return decorator

def with_presets(preset_bases, reason=None):
    available_presets: Set[str] = set(preset_bases)

    def decorator(fn):

        def wrapper(*args: Any, spec: Spec, **kw: Any):
            if spec.config.PRESET_BASE not in available_presets:
                message: str = f"doesn't support this preset base: {spec.config.PRESET_BASE}."
                if reason is not None:
                    message = f'{message} Reason: {reason}'
                dump_skipping_message(message)
                return None
            return fn(*args, spec=spec, **kw)
        return wrapper
    return decorator
with_light_client: Callable = with_phases(LIGHT_CLIENT_TESTING_FORKS)
with_altair_and_later: Callable = with_all_phases_from(ALTAIR)
with_bellatrix_and_later: Callable = with_all_phases_from(BELLATRIX)
with_capella_and_later: Callable = with_all_phases_from(CAPELLA)
with_deneb_and_later: Callable = with_all_phases_from(DENEB)
with_electra_and_later: Callable = with_all_phases_from(ELECTRA)
with_whisk_and_later: Callable = with_all_phases_from(WHISK, all_phases=ALLOWED_TEST_RUNNER_FORKS)
with_fulu_and_later: Callable = with_all_phases_from(FULU, all_phases=ALLOWED_TEST_RUNNER_FORKS)

class quoted_str(str):
    pass

def _get_basic_dict(ssz_dict):
    result: Dict[str, Any] = {}
    for k, v in ssz_dict.items():
        if isinstance(v, int):
            value: Union[int, bytes, str] = int(v)
        elif isinstance(v, bytes):
            value = bytes(bytearray(v))
        else:
            value = quoted_str(v)
        result[k] = value
    return result

def get_copy_of_spec(spec):
    fork: str = spec.fork
    preset: str = spec.config.PRESET_BASE
    module_path: str = f'eth2spec.{fork}.{preset}'
    module_spec: Any = importlib.util.find_spec(module_path)
    module: Any = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.config = deepcopy(spec.config)
    return module

def spec_with_config_overrides(spec, config_overrides):
    config: Dict[str, Any] = spec.config._asdict()
    config.update(((k, config_overrides[k]) for k in config.keys() & config_overrides.keys()))
    config_types: Dict[str, Any] = spec.Configuration.__annotations__
    modified_config: Dict[str, Any] = {k: config_types[k](v) for k, v in config.items()}
    spec.config = spec.Configuration(**modified_config)
    output_config: Dict[str, Any] = _get_basic_dict(modified_config)
    return (spec, output_config)

def with_config_overrides(config_overrides, emitted_fork=None, emit=True):

    def decorator(fn):

        def wrapper(*args: Any, spec: Spec, **kw: Any):
            spec, output_config = spec_with_config_overrides(get_copy_of_spec(spec), config_overrides)
            if 'phases' in kw:
                phases: Dict[str, Any] = {}
                for fork in kw['phases']:
                    phases[fork], output = spec_with_config_overrides(get_copy_of_spec(kw['phases'][fork]), config_overrides)
                    if emitted_fork == fork:
                        output_config = output
                kw['phases'] = phases
            if emit:
                yield ('config', 'cfg', output_config)
            out: Any = fn(*args, spec=spec, **kw)
            if out is not None:
                yield from out
        return wrapper
    return decorator

def only_generator(reason):

    def _decorator(inner):

        def _wrapper(*args: Any, **kwargs: Any):
            if is_pytest:
                dump_skipping_message(reason)
                return None
            return inner(*args, **kwargs)
        return _wrapper
    return _decorator

def with_test_suite_name(suite_name):

    def _decorator(inner):
        inner.suite_name = suite_name
        return inner
    return _decorator

def set_fork_metas(fork_metas):

    def decorator(fn):

        def wrapper(*args: Any, **kwargs: Any):
            return fn(*args, fork_metas=fork_metas, **kwargs)
        return wrapper
    return decorator

def with_fork_metas(fork_metas):
    run_yield_fork_meta: Callable = yield_fork_meta(fork_metas)
    run_with_phases: Callable = with_phases(ALL_PHASES)
    run_set_fork_metas: Callable = set_fork_metas(fork_metas)

    def decorator(fn):
        return run_set_fork_metas(run_with_phases(spec_test(with_state(run_yield_fork_meta(fn)))))
    return decorator

def yield_fork_meta(fork_metas):

    def decorator(fn):

        def wrapper(*args: Any, **kw: Any):
            phases: Dict[str, Any] = kw.pop('phases')
            spec: Spec = kw['spec']
            try:
                fork_meta: ForkMeta = next(filter(lambda m: m.pre_fork_name == spec.fork, fork_metas))
            except StopIteration:
                dump_skipping_message(f"doesn't support this fork: {spec.fork}")
            post_spec: Spec = phases[fork_meta.post_fork_name]
            pre_fork_counter: int = 0

            def pre_tag(obj):
                nonlocal pre_fork_counter
                pre_fork_counter += 1
                return obj

            def post_tag(obj):
                return obj
            yield ('post_fork', 'meta', fork_meta.post_fork_name)
            has_fork_epoch: bool = False
            if fork_meta.fork_epoch:
                kw['fork_epoch'] = fork_meta.fork_epoch
                has_fork_epoch = True
                yield ('fork_epoch', 'meta', fork_meta.fork_epoch)
            result: Any = fn(*args, post_spec=post_spec, pre_tag=pre_tag, post_tag=post_tag, **kw)
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