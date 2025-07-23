from importlib import import_module
from inspect import getmembers, isfunction
from pkgutil import walk_packages
from typing import Any, Callable, Dict, Iterable, Optional, List, Union, Generator
from eth2spec.utils import bls
from eth2spec.test.helpers.constants import ALL_PRESETS, TESTGEN_FORKS
from eth2spec.test.helpers.typing import SpecForkName, PresetBaseName
from eth2spec.gen_helpers.gen_base import gen_runner
from eth2spec.gen_helpers.gen_base.gen_typing import TestCase, TestProvider

def generate_case_fn(tfn: Callable[..., Any], generator_mode: bool, phase: str, preset: str, bls_active: bool) -> Callable[[], TestCase]:
    return lambda: tfn(generator_mode=generator_mode, phase=phase, preset=preset, bls_active=bls_active)

def generate_from_tests(runner_name: str, handler_name: str, src: Any, fork_name: SpecForkName, preset_name: PresetBaseName, bls_active: bool = True, phase: Optional[str] = None) -> Generator[TestCase, None, None]:
    fn_names = [name for name, _ in getmembers(src, isfunction) if name.startswith('test_')]
    if phase is None:
        phase = fork_name
    print('generating test vectors from tests source: %s' % src.__name__)
    for name in fn_names:
        tfn = getattr(src, name)
        case_name = name
        if case_name.startswith('test_'):
            case_name = case_name[5:]
        yield TestCase(fork_name=fork_name, preset_name=preset_name, runner_name=runner_name, handler_name=handler_name, suite_name=getattr(tfn, 'suite_name', 'pyspec_tests'), case_name=case_name, case_fn=generate_case_fn(tfn, generator_mode=True, phase=phase, preset=preset_name, bls_active=bls_active))

def get_provider(create_provider_fn: Callable[..., TestProvider], fork_name: SpecForkName, preset_name: PresetBaseName, all_mods: Dict[str, Dict[str, Union[str, List[str]]]]) -> Generator[TestProvider, None, None]:
    for key, mod_name in all_mods[fork_name].items():
        if not isinstance(mod_name, List):
            mod_name = [mod_name]
        yield create_provider_fn(fork_name=fork_name, preset_name=preset_name, handler_name=key, tests_src_mod_name=mod_name)

def get_create_provider_fn(runner_name: str) -> Callable[..., TestProvider]:

    def prepare_fn() -> None:
        bls.use_fastest()
        return

    def create_provider(fork_name: SpecForkName, preset_name: PresetBaseName, handler_name: str, tests_src_mod_name: List[str]) -> TestProvider:

        def cases_fn() -> Generator[TestCase, None, None]:
            for mod_name in tests_src_mod_name:
                tests_src = import_module(mod_name)
                yield from generate_from_tests(runner_name=runner_name, handler_name=handler_name, src=tests_src, fork_name=fork_name, preset_name=preset_name)
        return TestProvider(prepare=prepare_fn, make_cases=cases_fn)
    return create_provider

def run_state_test_generators(runner_name: str, all_mods: Dict[str, Dict[str, Union[str, List[str]]]], presets: List[PresetBaseName] = ALL_PRESETS, forks: List[SpecForkName] = TESTGEN_FORKS) -> None:
    for preset_name in presets:
        for fork_name in forks:
            if fork_name in all_mods:
                gen_runner.run_generator(runner_name, get_provider(create_provider_fn=get_create_provider_fn(runner_name), fork_name=fork_name, preset_name=preset_name, all_mods=all_mods))

def combine_mods(dict_1: Dict[str, Union[str, List[str]]], dict_2: Dict[str, Union[str, List[str]]]) -> Dict[str, List[str]]:
    dict_3 = {**dict_1, **dict_2}
    intersection = dict_1.keys() & dict_2.keys()
    for key in intersection:
        if not isinstance(dict_3[key], List):
            dict_3[key] = [dict_3[key]]
        if isinstance(dict_1[key], List):
            dict_3[key] += dict_1[key]
        else:
            dict_3[key].append(dict_1[key])
    return dict_3

def check_mods(all_mods: Dict[str, Dict[str, Union[str, List[str]]]], pkg: str) -> None:

    def get_expected_modules(package: str, absolute: bool = False) -> List[str]:
        modules = []
        eth2spec = import_module('eth2spec')
        prefix = eth2spec.__name__ + '.'
        for _, modname, ispkg in walk_packages(eth2spec.__path__, prefix):
            s = package if absolute else f'.{package}.'
            if s in modname and (not ispkg):
                modules.append(modname)
        return modules

    mods = []
    for fork in all_mods:
        for mod in all_mods[fork].values():
            if isinstance(mod, str):
                mod = [mod]
            for sub in mod:
                is_package = '.test_' not in sub
                if is_package:
                    mods.extend(get_expected_modules(sub, absolute=True))
                else:
                    mods.append(sub)
    problems = []
    expected_mods = get_expected_modules(pkg)
    if mods != expected_mods:
        for e in expected_mods:
            fork = e.split('.')[2]
            if fork not in all_mods:
                continue
            if '.unittests.' in e:
                continue
            if e not in mods:
                problems.append('missing: ' + e)
        for t in mods:
            if t.startswith('eth2spec.test.helpers'):
                continue
            if t not in expected_mods:
                print('unexpected:', t)
                problems.append('unexpected: ' + t)
    if problems:
        raise Exception('[ERROR] module problems:\n ' + '\n '.join(problems))
