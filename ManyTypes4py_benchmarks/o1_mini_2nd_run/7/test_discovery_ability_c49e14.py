"""Statistical tests over the forms of the distributions in the standard set of
definitions.

These tests all take the form of a classic hypothesis test with the null
hypothesis being that the probability of some event occurring when
drawing data from the distribution produced by some specifier is >=
REQUIRED_P
"""
import collections
import math
import re
from typing import Callable, Optional, Tuple, TypeVar, Any

from hypothesis import HealthCheck, settings as Settings
from hypothesis.control import BuildContext
from hypothesis.errors import UnsatisfiedAssumption
from hypothesis.internal import reflection
from hypothesis.internal.conjecture.engine import ConjectureRunner
from hypothesis.strategies import (
    binary,
    booleans,
    floats,
    integers,
    just,
    lists,
    one_of,
    sampled_from,
    sets,
    text,
    tuples,
    Strategy,
)
from tests.common.utils import no_shrink

T = TypeVar("T")

RUNS: int = 100
INITIAL_LAMBDA: re.Pattern = re.compile("^lambda[^:]*:\\s*")


def strip_lambda(s: str) -> str:
    return INITIAL_LAMBDA.sub("", s)


class HypothesisFalsified(AssertionError):
    pass


def define_test(
    specifier: Strategy[T],
    predicate: Callable[[T], bool],
    condition: Optional[Callable[[T], bool]] = None,
    p: float = 0.5,
    suppress_health_check: Tuple[HealthCheck, ...] = (),
) -> Callable[[], None]:
    required_runs: int = int(RUNS * p)

    def run_test() -> None:
        if condition is None:

            def _condition(x: T) -> bool:
                return True

            condition_string: str = ""
        else:
            _condition = condition
            condition_string = strip_lambda(
                reflection.get_pretty_function_description(condition)
            )

        def test_function(data: BuildContext) -> None:
            with BuildContext(data):
                try:
                    value: T = data.draw(specifier)
                except UnsatisfiedAssumption:
                    data.mark_invalid()
                    return
                if not _condition(value):
                    data.mark_invalid()
                if predicate(value):
                    data.mark_interesting()

        successes: int = 0
        actual_runs: int = 0
        for actual_runs in range(1, RUNS + 1):
            runner: ConjectureRunner = ConjectureRunner(
                test_function,
                settings=Settings(
                    max_examples=150,
                    phases=no_shrink,
                    suppress_health_check=suppress_health_check,
                ),
            )
            runner.run()
            if runner.interesting_examples:
                successes += 1
                if successes >= required_runs:
                    return
            if required_runs - successes > RUNS - actual_runs:
                break
        event: str = reflection.get_pretty_function_description(predicate)
        if condition is not None:
            event += "|"
            event += condition_string
        raise HypothesisFalsified(
            f"P({event}) ~ {successes} / {actual_runs} = {successes / actual_runs:.2f} < {required_runs / RUNS:.2f}; rejected"
        )

    return run_test


test_can_produce_zero: Callable[[], None] = define_test(integers(), lambda x: x == 0)
test_can_produce_large_magnitude_integers: Callable[[], None] = define_test(
    integers(), lambda x: abs(x) > 1000
)
test_can_produce_large_positive_integers: Callable[[], None] = define_test(
    integers(), lambda x: x > 1000
)
test_can_produce_large_negative_integers: Callable[[], None] = define_test(
    integers(), lambda x: x < -1000
)


def long_list(xs: list[Any]) -> bool:
    return len(xs) >= 10


test_can_produce_unstripped_strings: Callable[[], None] = define_test(
    text(), lambda x: x != x.strip()
)
test_can_produce_stripped_strings: Callable[[], None] = define_test(
    text(), lambda x: x == x.strip()
)
test_can_produce_multi_line_strings: Callable[[], None] = define_test(
    text(), lambda x: "\n" in x, p=0.35
)
test_can_produce_ascii_strings: Callable[[], None] = define_test(
    text(), lambda x: all((ord(c) <= 127 for c in x))
)
test_can_produce_long_strings_with_no_ascii: Callable[[], None] = define_test(
    text(min_size=5), lambda x: all((ord(c) > 127 for c in x)), p=0.1
)
test_can_produce_short_strings_with_some_non_ascii: Callable[[], None] = define_test(
    text(),
    lambda x: any((ord(c) > 127 for c in x)),
    condition=lambda x: len(x) <= 3,
)
test_can_produce_large_binary_strings: Callable[[], None] = define_test(
    binary(), lambda x: len(x) > 10, p=0.3
)
test_can_produce_positive_infinity: Callable[[], None] = define_test(
    floats(), lambda x: x == math.inf
)
test_can_produce_negative_infinity: Callable[[], None] = define_test(
    floats(), lambda x: x == -math.inf
)
test_can_produce_nan: Callable[[], None] = define_test(
    floats(), math.isnan
)
test_can_produce_floats_near_left: Callable[[], None] = define_test(
    floats(0, 1), lambda t: t < 0.2
)
test_can_produce_floats_near_right: Callable[[], None] = define_test(
    floats(0, 1), lambda t: t > 0.8
)
test_can_produce_floats_in_middle: Callable[[], None] = define_test(
    floats(0, 1), lambda t: 0.2 <= t <= 0.8
)
test_can_produce_long_lists: Callable[[], None] = define_test(
    lists(integers()), long_list, p=0.3
)
test_can_produce_short_lists: Callable[[], None] = define_test(
    lists(integers()), lambda x: len(x) <= 10
)
test_can_produce_the_same_int_twice: Callable[[], None] = define_test(
    lists(integers()), lambda t: len(set(t)) < len(t)
)


def distorted_value(x: list[type[Any]]) -> bool:
    c: collections.Counter = collections.Counter(x)
    return min(c.values()) * 3 <= max(c.values())


def distorted(x: list[type[Any]]) -> bool:
    return distorted_value(list(map(type, x)))


test_sampled_from_large_number_can_mix: Callable[[], None] = define_test(
    lists(sampled_from(range(50)), min_size=50), lambda x: len(set(x)) >= 25
)
test_sampled_from_often_distorted: Callable[[], None] = define_test(
    lists(sampled_from(range(5))),
    distorted_value,
    condition=lambda x: len(x) >= 3,
)
test_non_empty_subset_of_two_is_usually_large: Callable[[], None] = define_test(
    sets(sampled_from((1, 2))), lambda t: len(t) == 2
)
test_subset_of_ten_is_sometimes_empty: Callable[[], None] = define_test(
    sets(integers(1, 10)), lambda t: len(t) == 0
)
test_mostly_sensible_floats: Callable[[], None] = define_test(
    floats(), lambda t: t + 1 > t
)
test_mostly_largish_floats: Callable[[], None] = define_test(
    floats(), lambda t: t + 1 > 1, condition=lambda x: x > 0
)
test_ints_can_occasionally_be_really_large: Callable[[], None] = define_test(
    integers(), lambda t: t >= 2 ** 63
)
test_mixing_is_sometimes_distorted: Callable[[], None] = define_test(
    lists(booleans() | tuples()),
    distorted,
    condition=lambda x: len(set(map(type, x))) == 2,
    suppress_health_check=(HealthCheck.filter_too_much,),
)
test_mixes_2_reasonably_often: Callable[[], None] = define_test(
    lists(booleans() | tuples()),
    lambda x: len(set(map(type, x))) > 1,
    condition=bool,
)
test_partial_mixes_3_reasonably_often: Callable[[], None] = define_test(
    lists(booleans() | tuples() | just("hi")),
    lambda x: 1 < len(set(map(type, x))) < 3,
    condition=bool,
)
test_mixes_not_too_often: Callable[[], None] = define_test(
    lists(booleans() | tuples()),
    lambda x: len(set(map(type, x))) == 1,
    condition=bool,
)
test_integers_are_usually_non_zero: Callable[[], None] = define_test(
    integers(), lambda x: x != 0
)
test_integers_are_sometimes_zero: Callable[[], None] = define_test(
    integers(), lambda x: x == 0
)
test_integers_are_often_small: Callable[[], None] = define_test(
    integers(), lambda x: abs(x) <= 100
)
test_integers_are_often_small_but_not_that_small: Callable[[], None] = define_test(
    integers(), lambda x: 50 <= abs(x) <= 255
)

one_of_nested_strategy: Strategy[int] = one_of(
    just(0),
    one_of(
        just(1),
        just(2),
        one_of(just(3), just(4), one_of(just(5), just(6), just(7))),
    ),
)

for i in range(8):
    exec(
        f"""test_one_of_flattens_branches_{i}: Callable[[], None] = define_test(
        one_of_nested_strategy, lambda x: x == {i}
    )"""
    )

xor_nested_strategy: Strategy[Any] = just(0) | (
    just(1)
    | just(2)
    | (just(3) | just(4) | (just(5) | just(6) | just(7)))
)

for i in range(8):
    exec(
        f"""test_xor_flattens_branches_{i}: Callable[[], None] = define_test(
        xor_nested_strategy, lambda x: x == {i}
    )"""
    )


def double(x: int) -> int:
    return x * 2


one_of_nested_strategy_with_map: Strategy[int] = one_of(
    just(1),
    one_of(
        (just(2) | just(3)).map(double),
        one_of(
            (just(4) | just(5)).map(double),
            one_of(
                (just(6) | just(7) | just(8)).map(double)
            ).map(double),
        ),
    ),
)

for i in (1, 4, 6, 16, 20, 24, 28, 32):
    exec(
        f"""test_one_of_flattens_map_branches_{i}: Callable[[], None] = define_test(
        one_of_nested_strategy_with_map, lambda x: x == {i}
    )"""
    )

one_of_nested_strategy_with_flatmap: Strategy[list[Any]] = one_of(
    just(None)
).flatmap(
    lambda x: one_of(
        just([x] * 0),
        just([x] * 1),
        one_of(
            just([x] * 2),
            just([x] * 3),
            one_of(just([x] * 4), just([x] * 5), one_of(just([x] * 6), just([x] * 7))),
        ),
    )
)

for i in range(8):
    exec(
        f"""test_one_of_flattens_flatmap_branches_{i}: Callable[[], None] = define_test(
        one_of_nested_strategy_with_flatmap, lambda x: len(x) == {i}
    )"""
    )

xor_nested_strategy_with_flatmap: Strategy[list[Any]] = just(None).flatmap(
    lambda x: just([x] * 0)
    | just([x] * 1)
    | (just([x] * 2) | just([x] * 3) | (just([x] * 4) | just([x] * 5) | (just([x] * 6) | just([x] * 7))))
)

for i in range(8):
    exec(
        f"""test_xor_flattens_flatmap_branches_{i}: Callable[[], None] = define_test(
        xor_nested_strategy_with_flatmap, lambda x: len(x) == {i}
    )"""
    )

one_of_nested_strategy_with_filter: Strategy[int] = one_of(
    just(0),
    just(1),
    one_of(
        just(2),
        just(3),
        one_of(just(4), just(5), one_of(just(6), just(7))),
    ),
).filter(lambda x: x % 2 == 0)

for i in range(4):
    exec(
        f"""test_one_of_flattens_filter_branches_{i}: Callable[[], None] = define_test(
        one_of_nested_strategy_with_filter, lambda x: x == 2 * {i}
    )"""
    )

test_long_duplicates_strings: Callable[[], None] = define_test(
    tuples(text(), text()), lambda s: len(s[0]) >= 5 and s[0] == s[1]
)
