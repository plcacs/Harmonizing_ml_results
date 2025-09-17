import hashlib
import math
from random import Random
from typing import Any, List, Optional, Tuple, Union, TypeVar, Sequence

from hypothesis import Verbosity, assume, settings
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.internal.compat import PYPY
from hypothesis.internal.floats import clamp, float_to_int, int_to_float, is_negative
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule
from hypothesis.strategies import (
    binary,
    booleans,
    complex_numbers,
    data,
    decimals,
    floats,
    fractions,
    integers,
    just,
    lists,
    none,
    sampled_from,
    text,
    tuples,
    SearchStrategy,
)

T = TypeVar('T')
U = TypeVar('U')

AVERAGE_LIST_LENGTH: int = 2

class HypothesisSpec(RuleBasedStateMachine):
    database: Optional[InMemoryExampleDatabase]

    strategies: Bundle[Any] = Bundle('strategy')
    strategy_tuples: Bundle[Any] = Bundle('tuples')
    objects: Bundle[Any] = Bundle('objects')
    basic_data: Bundle[Any] = Bundle('basic')
    varied_floats: Bundle[Any] = Bundle('varied_floats')

    def __init__(self) -> None:
        super().__init__()
        self.database = None

    def teardown(self) -> None:
        self.clear_database()

    @rule()
    def clear_database(self) -> None:
        if self.database is not None:
            self.database = None

    @rule()
    def set_database(self) -> None:
        self.teardown()
        self.database = InMemoryExampleDatabase()

    @rule(target=strategies, spec=sampled_from(
        (integers(), booleans(), floats(), complex_numbers(), fractions(), decimals(), text(), binary(), none(), tuples())
    ))
    def strategy(self, spec: SearchStrategy[Any]) -> SearchStrategy[Any]:
        return spec

    @rule(target=strategies, values=lists(integers() | text(), min_size=1))
    def sampled_from_strategy(self, values: List[Union[int, str]]) -> SearchStrategy[Union[int, str]]:
        return sampled_from(values)

    @rule(target=strategies, spec=strategy_tuples)
    def strategy_for_tupes(self, spec: Tuple[Any, ...]) -> SearchStrategy[Any]:
        # This returns a tuple strategy using the unpacked spec.
        return tuples(*spec)

    @rule(target=strategies, source=strategies, level=integers(1, 10), mixer=text())
    def filtered_strategy(self, source: SearchStrategy[T], level: int, mixer: str) -> SearchStrategy[T]:
        def is_good(x: Any) -> bool:
            seed: bytes = hashlib.sha384((mixer + repr(x)).encode()).digest()
            return bool(Random(seed).randint(0, level))
        return source.filter(is_good)

    @rule(target=strategies, elements=strategies)
    def list_strategy(self, elements: SearchStrategy[T]) -> SearchStrategy[List[T]]:
        return lists(elements)

    @rule(target=strategies, left=strategies, right=strategies)
    def or_strategy(self, left: SearchStrategy[T], right: SearchStrategy[T]) -> SearchStrategy[T]:
        return left | right

    @rule(target=varied_floats, source=floats())
    def float(self, source: float) -> float:
        return source

    @rule(target=varied_floats, source=varied_floats, offset=integers(-100, 100))
    def adjust_float(self, source: float, offset: int) -> float:
        return int_to_float(clamp(0, float_to_int(source) + offset, 2 ** 64 - 1))

    @rule(target=strategies, left=varied_floats, right=varied_floats)
    def float_range(self, left: float, right: float) -> SearchStrategy[float]:
        assume(math.isfinite(left) and math.isfinite(right))
        left, right = sorted((left, right))
        assert left <= right
        assume(left or right or (not (is_negative(right) and (not is_negative(left)))))
        return floats(left, right)

    @rule(target=strategies, source=strategies, result1=strategies, result2=strategies, mixer=text(), p=floats(0, 1))
    def flatmapped_strategy(
        self,
        source: SearchStrategy[T],
        result1: SearchStrategy[U],
        result2: SearchStrategy[U],
        mixer: str,
        p: float,
    ) -> SearchStrategy[U]:
        assume(result1 is not result2)

        def do_map(value: T) -> SearchStrategy[U]:
            rep: str = repr(value)
            rnd: Random = Random(hashlib.sha384((mixer + rep).encode()).digest())
            if rnd.random() <= p:
                return result1
            else:
                return result2

        return source.flatmap(do_map)

    @rule(target=strategies, value=objects)
    def just_strategy(self, value: Any) -> SearchStrategy[Any]:
        return just(value)

    @rule(target=strategy_tuples, source=strategies)
    def single_tuple(self, source: SearchStrategy[Any]) -> Tuple[SearchStrategy[Any]]:
        return (source,)

    @rule(target=strategy_tuples, left=strategy_tuples, right=strategy_tuples)
    def cat_tuples(
        self, left: Tuple[SearchStrategy[Any], ...], right: Tuple[SearchStrategy[Any], ...]
    ) -> Tuple[SearchStrategy[Any], ...]:
        return left + right

    @rule(target=objects, strat=strategies, data=data())
    def get_example(self, strat: SearchStrategy[Any], data: Any) -> None:
        data.draw(strat)

    @rule(target=strategies, left=integers(), right=integers())
    def integer_range(self, left: int, right: int) -> SearchStrategy[int]:
        left, right = sorted((left, right))
        return integers(left, right)

    @rule(strat=strategies)
    def repr_is_good(self, strat: SearchStrategy[Any]) -> None:
        assert ' at 0x' not in repr(strat)


MAIN: bool = __name__ == '__main__'
TestHypothesis = HypothesisSpec.TestCase  # type: ignore
TestHypothesis.settings = settings(
    TestHypothesis.settings,
    stateful_step_count=10 if PYPY else 50,
    verbosity=max(TestHypothesis.settings.verbosity, Verbosity.verbose),
    max_examples=10000 if MAIN else 200,
)
if MAIN:
    TestHypothesis().runTest()