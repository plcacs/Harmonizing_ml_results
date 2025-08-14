# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import hashlib
import math
from random import Random
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from hypothesis import Verbosity, assume, settings
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.internal.compat import PYPY
from hypothesis.internal.floats import clamp, float_to_int, int_to_float, is_negative
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule
from hypothesis.strategies import (
    SearchStrategy,
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
)

AVERAGE_LIST_LENGTH = 2


class HypothesisSpec(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.database: Optional[InMemoryExampleDatabase] = None

    strategies: Bundle[SearchStrategy[Any]] = Bundle("strategy")
    strategy_tuples: Bundle[Tuple[SearchStrategy[Any], ...]] = Bundle("tuples")
    objects: Bundle[Any] = Bundle("objects")
    basic_data: Bundle[Any] = Bundle("basic")
    varied_floats: Bundle[float] = Bundle("varied_floats")

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

    @rule(
        target=strategies,
        spec=sampled_from(
            (
                integers(),
                booleans(),
                floats(),
                complex_numbers(),
                fractions(),
                decimals(),
                text(),
                binary(),
                none(),
                tuples(),
            )
        ),
    )
    def strategy(self, spec: SearchStrategy[Any]) -> SearchStrategy[Any]:
        return spec

    @rule(target=strategies, values=lists(integers() | text(), min_size=1))
    def sampled_from_strategy(self, values: List[Union[int, str]]) -> SearchStrategy[Any]:
        return sampled_from(values)

    @rule(target=strategies, spec=strategy_tuples)
    def strategy_for_tupes(self, spec: Tuple[SearchStrategy[Any], ...) -> SearchStrategy[Any]:
        return tuples(*spec)

    @rule(target=strategies, source=strategies, level=int, mixer=str)
    def filtered_strategy(
        self, source: SearchStrategy[Any], level: int, mixer: str
    ) -> SearchStrategy[Any]:
        def is_good(x: Any) -> bool:
            seed = hashlib.sha384((mixer + repr(x)).encode()).digest()
            return bool(Random(seed).randint(0, level))

        return source.filter(is_good)

    @rule(target=strategies, elements=strategies)
    def list_strategy(self, elements: SearchStrategy[Any]) -> SearchStrategy[List[Any]]:
        return lists(elements)

    @rule(target=strategies, left=strategies, right=strategies)
    def or_strategy(
        self, left: SearchStrategy[Any], right: SearchStrategy[Any]
    ) -> SearchStrategy[Any]:
        return left | right

    @rule(target=varied_floats, source=floats())
    def float(self, source: float) -> float:
        return source

    @rule(target=varied_floats, source=varied_floats, offset=int)
    def adjust_float(self, source: float, offset: int) -> float:
        return int_to_float(clamp(0, float_to_int(source) + offset, 2**64 - 1))

    @rule(target=strategies, left=varied_floats, right=varied_floats)
    def float_range(self, left: float, right: float) -> SearchStrategy[float]:
        assume(math.isfinite(left) and math.isfinite(right))
        left, right = sorted((left, right))
        assert left <= right
        # exclude deprecated case where left = 0.0 and right = -0.0
        assume(left or right or not (is_negative(right) and not is_negative(left)))
        return floats(left, right)

    @rule(
        target=strategies,
        source=strategies,
        result1=strategies,
        result2=strategies,
        mixer=str,
        p=float,
    )
    def flatmapped_strategy(
        self,
        source: SearchStrategy[Any],
        result1: SearchStrategy[Any],
        result2: SearchStrategy[Any],
        mixer: str,
        p: float,
    ) -> SearchStrategy[Any]:
        assume(result1 is not result2)

        def do_map(value: Any) -> SearchStrategy[Any]:
            rep = repr(value)
            random = Random(hashlib.sha384((mixer + rep).encode()).digest())
            if random.random() <= p:
                return result1
            else:
                return result2

        return source.flatmap(do_map)

    @rule(target=strategies, value=objects)
    def just_strategy(self, value: Any) -> SearchStrategy[Any]:
        return just(value)

    @rule(target=strategy_tuples, source=strategies)
    def single_tuple(self, source: SearchStrategy[Any]) -> Tuple[SearchStrategy[Any], ...]:
        return (source,)

    @rule(target=strategy_tuples, left=strategy_tuples, right=strategy_tuples)
    def cat_tuples(
        self,
        left: Tuple[SearchStrategy[Any], ...],
        right: Tuple[SearchStrategy[Any], ...],
    ) -> Tuple[SearchStrategy[Any], ...]:
        return left + right

    @rule(target=objects, strat=strategies, data=data())
    def get_example(self, strat: SearchStrategy[Any], data: Any) -> Any:
        return data.draw(strat)

    @rule(target=strategies, left=int, right=int)
    def integer_range(self, left: int, right: int) -> SearchStrategy[int]:
        left, right = sorted((left, right))
        return integers(left, right)

    @rule(strat=strategies)
    def repr_is_good(self, strat: SearchStrategy[Any]) -> None:
        assert " at 0x" not in repr(strat)


MAIN: bool = __name__ == "__main__"

TestHypothesis = HypothesisSpec.TestCase

TestHypothesis.settings = settings(
    TestHypothesis.settings,
    stateful_step_count=10 if PYPY else 50,
    verbosity=max(TestHypothesis.settings.verbosity, Verbosity.verbose),
    max_examples=10000 if MAIN else 200,
)

if MAIN:
    TestHypothesis().runTest()
