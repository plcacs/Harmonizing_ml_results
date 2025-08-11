from importlib import import_module
from inspect import getmembers, isfunction
from pkgutil import walk_packages
from typing import Any, Callable, Dict, Iterable, Optional, List, Union
from eth2spec.utils import bls
from eth2spec.test.helpers.constants import ALL_PRESETS, TESTGEN_FORKS
from eth2spec.test.helpers.typing import SpecForkName, PresetBaseName
from eth2spec.gen_helpers.gen_base import gen_runner
from eth2spec.gen_helpers.gen_base.gen_typing import TestCase, TestProvider

def generate_case_fn(tfn: bool, generator_mode: bool, phase: bool, preset: bool, bls_active: bool) -> typing.Callable:
    return lambda: tfn(generator_mode=generator_mode, phase=phase, preset=preset, bls_active=bls_active)

def generate_from_tests(runner_name: Union[str, dict], handler_name: Union[str, dict], src: Union[str, None], fork_name: Union[str, dict[str, typing.Any], resttesmodels.TestCase], preset_name: Union[str, dict], bls_active: bool=True, phase: Union[None, str]=None) -> typing.Generator[TestCase]:
    """
    Generate a list of test cases by running tests from the given src in generator-mode.
    :param runner_name: to categorize the test in general as.
    :param handler_name: to categorize the test specialization as.
    :param src: to retrieve tests from (discovered using inspect.getmembers).
    :param fork_name: the folder name for these tests.
           (if multiple forks are applicable, indicate the last fork)
    :param preset_name: to select a preset. Tests that do not support the preset will be skipped.
    :param bls_active: optional, to override BLS switch preference. Defaults to True.
    :param phase: optional, to run tests against a particular spec version. Default to `fork_name` value.
           Set to the pre-fork (w.r.t. fork_name) in multi-fork tests.
    :return: an iterable of test cases.
    """
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

def get_provider(create_provider_fn: Union[str, bool, dict[str, typing.Any]], fork_name: Union[dict, str, typing.Callable], preset_name: Union[str, bool, dict[str, typing.Any]], all_mods: Union[dict, dict[str, typing.Any], str]) -> typing.Generator:
    for key, mod_name in all_mods[fork_name].items():
        if not isinstance(mod_name, List):
            mod_name = [mod_name]
        yield create_provider_fn(fork_name=fork_name, preset_name=preset_name, handler_name=key, tests_src_mod_name=mod_name)

def get_create_provider_fn(runner_name: Union[str, typing.Sequence[str]]):

    def prepare_fn() -> None:
        bls.use_fastest()
        return

    def create_provider(fork_name: Any, preset_name: Any, handler_name: Any, tests_src_mod_name: Any) -> TestProvider:

        def cases_fn() -> typing.Generator:
            for mod_name in tests_src_mod_name:
                tests_src = import_module(mod_name)
                yield from generate_from_tests(runner_name=runner_name, handler_name=handler_name, src=tests_src, fork_name=fork_name, preset_name=preset_name)
        return TestProvider(prepare=prepare_fn, make_cases=cases_fn)
    return create_provider

def run_state_test_generators(runner_name: Union[str, cmk.utils.type_defs.HostName], all_mods: Union[str, typing.Callable], presets: Any=ALL_PRESETS, forks: Any=TESTGEN_FORKS) -> None:
    """
    Generate all available state tests of `TESTGEN_FORKS` forks of `ALL_PRESETS` presets of the given runner.
    """
    for preset_name in presets:
        for fork_name in forks:
            if fork_name in all_mods:
                gen_runner.run_generator(runner_name, get_provider(create_provider_fn=get_create_provider_fn(runner_name), fork_name=fork_name, preset_name=preset_name, all_mods=all_mods))

def combine_mods(dict_1: Union[dict, dict[str, list[typing.Any]]], dict_2: Union[dict, dict[str, list[typing.Any]]]) -> dict[typing.Union[dict,dict[str, list[typing.Any]],str], typing.Union[dict,dict[str, list[typing.Any]],list[typing.Union[dict,dict[str, list[typing.Any]]]]]]:
    """
    Return the merged dicts, where the result value would be a list of the values from two dicts.
    """
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

def check_mods(all_mods: list[str], pkg: str) -> None:
    """
    Raise an exception if there is a missing/unexpected module in all_mods.
    """

    def get_expected_modules(package: Any, absolute: bool=False) -> list:
        """
        Return all modules (which are not packages) inside the given package.
        """
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