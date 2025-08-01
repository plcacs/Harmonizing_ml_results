from random import Random
from typing import Any, Callable, Iterable, Iterator, List, Tuple, Type
from inspect import getmembers, isclass

from eth2spec.gen_helpers.gen_base import gen_runner, gen_typing
from eth2spec.debug import random_value, encode
from eth2spec.test.helpers.constants import TESTGEN_FORKS, MINIMAL, MAINNET
from eth2spec.test.context import spec_targets
from eth2spec.utils.ssz.ssz_typing import Container
from eth2spec.utils.ssz.ssz_impl import hash_tree_root, serialize

MAX_BYTES_LENGTH: int = 1000
MAX_LIST_LENGTH: int = 10


def create_test_case(
    rng: Random,
    typ: Type[Container],
    mode: random_value.RandomizationMode,
    chaos: bool
) -> Iterable[Tuple[str, str, Any]]:
    value = random_value.get_random_ssz_object(rng, typ, MAX_BYTES_LENGTH, MAX_LIST_LENGTH, mode, chaos)
    yield ('value', 'data', encode.encode(value))
    yield ('serialized', 'ssz', serialize(value))
    roots_data = {'root': '0x' + hash_tree_root(value).hex()}
    yield ('roots', 'data', roots_data)


def get_spec_ssz_types(spec: Any) -> List[Tuple[str, Type[Container]]]:
    return [
        (name, value)
        for name, value in getmembers(spec, isclass)
        if issubclass(value, Container) and value != Container
    ]


def ssz_static_cases(
    fork_name: str,
    preset_name: str,
    seed: int,
    name: str,
    ssz_type: Type[Container],
    mode: random_value.RandomizationMode,
    chaos: bool,
    count: int
) -> Iterable[gen_typing.TestCase]:
    random_mode_name = mode.to_name()
    rng: Random = Random(seed)
    for i in range(count):
        yield gen_typing.TestCase(
            fork_name=fork_name,
            preset_name=preset_name,
            runner_name='ssz_static',
            handler_name=name,
            suite_name=f'ssz_{random_mode_name}{("_chaos" if chaos else "")}',
            case_name=f'case_{i}',
            case_fn=lambda: create_test_case(rng, ssz_type, mode, chaos)  # type: Callable[[], Iterable[Tuple[str, str, Any]]]
        )


def create_provider(
    fork_name: str,
    preset_name: str,
    seed: int,
    mode: random_value.RandomizationMode,
    chaos: bool,
    cases_if_random: int
) -> gen_typing.TestProvider:

    def prepare_fn() -> None:
        return

    def cases_fn() -> Iterator[gen_typing.TestCase]:
        count: int = cases_if_random if chaos or mode.is_changing() else 1
        spec = spec_targets[preset_name][fork_name]
        for i, (name, ssz_type) in enumerate(get_spec_ssz_types(spec)):
            yield from ssz_static_cases(fork_name, preset_name, seed * 1000 + i, name, ssz_type, mode, chaos, count)

    return gen_typing.TestProvider(prepare=prepare_fn, make_cases=cases_fn)


if __name__ == '__main__':
    settings: List[Tuple[int, str, random_value.RandomizationMode, bool, int]] = []
    seed: int = 1
    for mode in random_value.RandomizationMode:
        settings.append((seed, MINIMAL, mode, False, 30))
        seed += 1
    settings.append((seed, MINIMAL, random_value.RandomizationMode.mode_random, True, 30))
    seed += 1
    settings.append((seed, MAINNET, random_value.RandomizationMode.mode_random, False, 5))
    seed += 1
    for fork in TESTGEN_FORKS:
        providers: List[gen_typing.TestProvider] = [
            create_provider(fork, preset_name, seed, mode, chaos, cases_if_random)
            for seed, preset_name, mode, chaos, cases_if_random in settings
        ]
        gen_runner.run_generator('ssz_static', providers)