#!/usr/bin/env python3
"""
This file tests for our ability to make precise shrinks.

Terminology: A shrink is *precise* if there is a single example (draw call) that
it replaces, without leaving any of the data before or after that draw call changed.
Otherwise, it is sloppy.

Precise shrinks correspond to the changes we can make to drawn data in isolation
of the rest of the test case. e.g. if we draw a list, we want to always be able
to delete an element from it without affecting things outside that list. If we
draw an integer, we always want to be able to subtract from it.

An example of a sloppy shrink is that we can sloppily replace any list with a prefix
of it by changing the boolean that says if we should draw more elements with False.
However leaves all the data corresponding to the rest of the list after that prefix
in the test case, so everything after the drawn list is deleted.

Having a rich vocabulary of precise shrinks we can make allows us to more easily
reason about shrunk data, because we can look at the data and think in terms of
what changes the shrinker would have made to it, and the fact that it hasn't
means we know that it's important. e.g. this numeric value can't be smaller, this
list can't have fewer elements.

Sloppy shrinks in contrast just make the test case smaller. This is still good,
obviously, and we rely on sloppy shrinks for a lot of shrinker performance and
quality - often what we can expect is that we get to a smaller test case faster
through sloppy shrinks, and then precise shrinks guarantee properties of the final
result.
"""

import itertools
from functools import lru_cache
from random import Random
from typing import Any, Callable, List, Optional, Tuple, TypeVar
import pytest
from hypothesis import find, settings, strategies as st
from hypothesis.control import BuildContext
from hypothesis.errors import StopTest, UnsatisfiedAssumption
from hypothesis.internal.conjecture.data import ConjectureData, ConjectureResult, Status
from hypothesis.internal.conjecture.engine import ConjectureRunner, ExitReason, RunIsComplete
from hypothesis.internal.conjecture.shrinker import sort_key

T = TypeVar("T")

def safe_draw(data: ConjectureData, strategy: st.SearchStrategy[T]) -> T:
    """Set up just enough of the Hypothesis machinery to use draw on
    a strategy."""
    with BuildContext(data):
        try:
            return data.draw(strategy)
        except UnsatisfiedAssumption:
            data.mark_invalid()
            # Although marking invalid, we need to return something.
            # This line will typically not be reached.
            raise

def precisely_shrink(
    strategy: st.SearchStrategy[T],
    is_interesting: Callable[[T], bool] = lambda x: True,
    initial_condition: Callable[[T], bool] = lambda x: True,
    end_marker: st.SearchStrategy[Any] = st.integers(),
    seed: int = 0
) -> Tuple[Any, T]:
    """Generates a random value from the strategy and then precisely shrinks it,
    by shrinking it with some value immediately afterwards that is not allowed to
    be modified during shrinking."""
    rnd: Random = Random(seed)
    while True:
        data: ConjectureData = ConjectureData(random=rnd)
        try:
            initial_value: T = safe_draw(data, strategy)
        except StopTest:
            continue
        if is_interesting(initial_value) and initial_condition(initial_value):
            break
    target_check_value: Any = safe_draw(data, end_marker)
    initial_choices: List[Any] = data.choices
    replay: ConjectureData = ConjectureData.for_choices(initial_choices)
    assert safe_draw(replay, strategy) == initial_value
    assert safe_draw(replay, end_marker) == target_check_value

    def test_function(local_data: ConjectureData) -> None:
        value: T = safe_draw(local_data, strategy)
        check_value: Any = safe_draw(local_data, end_marker)
        if is_interesting(value) and check_value == target_check_value:
            local_data.mark_interesting()

    runner: ConjectureRunner = ConjectureRunner(test_function, random=rnd)
    try:
        buf: ConjectureResult = runner.cached_test_function_ir(initial_choices)
        assert buf.status == Status.INTERESTING
        assert buf.choices == initial_choices
        assert runner.interesting_examples
        runner.shrink_interesting_examples()
    except RunIsComplete:
        assert runner.exit_reason in (ExitReason.finished, ExitReason.max_shrinks)
    result_example = next(iter(runner.interesting_examples.values()))
    data = ConjectureData.for_choices(result_example.choices)
    result_value: T = safe_draw(data, strategy)
    data.freeze()
    return (data.as_result(), result_value)

common_strategies_with_types: List[Tuple[type, st.SearchStrategy[Any]]] = [
    (type(None), st.none()),
    (bool, st.booleans()),
    (bytes, st.binary()),
    (str, st.text()),
    (int, st.integers()),
]
common_strategies: List[st.SearchStrategy[Any]] = [v for _, v in common_strategies_with_types]

@lru_cache(maxsize=None)
def minimal_for_strategy(s: st.SearchStrategy[Any]) -> Tuple[Any, Any]:
    return precisely_shrink(s, end_marker=st.none())

def minimal_nodes_for_strategy(s: st.SearchStrategy[Any]) -> Any:
    return minimal_for_strategy(s)[0].nodes

def test_strategy_list_is_in_sorted_order() -> None:
    assert common_strategies == sorted(common_strategies, key=lambda s: sort_key(minimal_nodes_for_strategy(s)))

@pytest.mark.parametrize("typ,strat", common_strategies_with_types)
@pytest.mark.parametrize("require_truthy", [False, True])
def test_can_precisely_shrink_values(typ: type, strat: st.SearchStrategy[Any], require_truthy: bool) -> None:
    if typ is type(None) and require_truthy:
        pytest.skip("None is falsey")
    cond: Callable[[Any], bool] = bool if require_truthy else (lambda x: True)
    result, shrunk: Any = precisely_shrink(strat, is_interesting=cond)
    assert shrunk == find(strat, cond)

AlternativesType = List[Tuple[Tuple[type, st.SearchStrategy[Any]], ...]]
alternatives: AlternativesType = [comb for n in (2, 3, 4) for comb in itertools.combinations(common_strategies_with_types, n)]
indexed_alternatives: List[Tuple[int, int, Tuple[Tuple[type, st.SearchStrategy[Any]], ...]]] = [
    (i, j, a) for a in alternatives for i, j in itertools.combinations(range(len(a)), 2)
]

@pytest.mark.parametrize("i,j,a", indexed_alternatives)
@pytest.mark.parametrize("seed", [0, 4389048901])
def test_can_precisely_shrink_alternatives(i: int, j: int, a: Tuple[Tuple[type, st.SearchStrategy[Any]], ...], seed: int) -> None:
    types: List[type] = [u for u, _ in a]
    combined_strategy: st.SearchStrategy[Any] = st.one_of(*[v for _, v in a])
    result, value = precisely_shrink(
        combined_strategy,
        initial_condition=lambda x: isinstance(x, types[j]),
        is_interesting=lambda x: not any(isinstance(x, types[k]) for k in range(i)),
        seed=seed
    )
    assert isinstance(value, types[i])

@pytest.mark.parametrize("a", list(itertools.combinations(common_strategies_with_types, 3)))
@pytest.mark.parametrize("seed", [0, 4389048901])
def test_precise_shrink_with_blocker(a: Tuple[Tuple[type, st.SearchStrategy[Any]], ...], seed: int) -> None:
    x, y, z = a
    a_reordered: Tuple[Tuple[type, st.SearchStrategy[Any]], ...] = (x, z, y)
    types: List[type] = [u for u, _ in a_reordered]
    combined_strategy: st.SearchStrategy[Any] = st.one_of(*[v for _, v in a_reordered])
    result, value = precisely_shrink(
        combined_strategy,
        initial_condition=lambda x: isinstance(x, types[2]),
        is_interesting=lambda x: True,
        seed=seed
    )
    assert isinstance(value, types[0])

def find_random(s: st.SearchStrategy[T], condition: Callable[[T], bool], seed: Optional[int] = None) -> Tuple[Any, T]:
    rnd: Random = Random(seed)
    while True:
        data: ConjectureData = ConjectureData(random=rnd)
        try:
            with BuildContext(data=data):
                value: T = data.draw(s)
                if condition(value):
                    data.freeze()
                    return (data.as_result(), value)
        except (StopTest, UnsatisfiedAssumption):
            continue

def shrinks(strategy: st.SearchStrategy[T], nodes: Any, *, allow_sloppy: bool = True, seed: int = 0) -> List[Tuple[Any, T]]:
    results: Dict[Any, T] = {}
    rnd: Random = Random(seed)
    choices: Tuple[Any, ...] = tuple(n.value for n in nodes)
    if allow_sloppy:
        def test_function(data: ConjectureData) -> None:
            value: T = safe_draw(data, strategy)
            results[data.nodes] = value
        runner: ConjectureRunner = ConjectureRunner(test_function, settings=settings(max_examples=10 ** 9))
        initial: ConjectureResult = runner.cached_test_function_ir(choices)
        assert isinstance(initial, ConjectureResult)
        try:
            runner.shrink(initial, lambda x: x.choices == initial.choices)
        except RunIsComplete:
            assert runner.exit_reason in (ExitReason.finished, ExitReason.max_shrinks)
    else:
        trial: ConjectureData = ConjectureData(prefix=choices, random=rnd)
        with BuildContext(trial):
            trial.draw(strategy)
            assert trial.choices == choices, "choice sequence is already sloppy"
            padding: Any = safe_draw(trial, st.integers())
        initial_choices: List[Any] = trial.choices

        def test_function(data: ConjectureData) -> None:
            value: T = safe_draw(data, strategy)
            key = data.nodes
            padding_check: Any = safe_draw(data, st.integers())
            if padding_check == padding:
                results[key] = value
        runner: ConjectureRunner = ConjectureRunner(test_function, settings=settings(max_examples=10 ** 9))
        initial: ConjectureResult = runner.cached_test_function_ir(initial_choices)
        assert len(results) == 1
        try:
            runner.shrink(initial, lambda x: x.choices == initial_choices)
        except RunIsComplete:
            assert runner.exit_reason in (ExitReason.finished, ExitReason.max_shrinks)
    results.pop(nodes)
    seen: set = set()
    result_list: List[Tuple[Any, T]] = []
    for k, v in sorted(results.items(), key=lambda x: sort_key(x[0])):
        t_repr = repr(v)
        if t_repr in seen:
            continue
        seen.add(t_repr)
        result_list.append((k, v))
    return result_list

@pytest.mark.parametrize("a", list(itertools.product(*[common_strategies[1:]] * 2)))
@pytest.mark.parametrize("block_falsey", [False, True])
@pytest.mark.parametrize("allow_sloppy", [False, True])
@pytest.mark.parametrize("seed", [0, 2452, 99085240570])
def test_always_shrinks_to_none(
    a: Tuple[st.SearchStrategy[Any], ...],
    seed: int,
    block_falsey: bool,
    allow_sloppy: bool
) -> None:
    combined_strategy: st.SearchStrategy[Any] = st.one_of(st.none(), *a)
    result, value = find_random(combined_strategy, lambda x: x is not None, seed=seed)
    shrunk_values: List[Tuple[Any, Any]] = shrinks(combined_strategy, result.nodes, allow_sloppy=allow_sloppy, seed=seed)
    assert shrunk_values[0][1] is None

@pytest.mark.parametrize("i,alts", [(i, alt) for alt in alternatives for i in range(1, len(alt))])
@pytest.mark.parametrize("force_small", [False, True])
@pytest.mark.parametrize("seed", [0, 2452, 99085240570])
def test_can_shrink_to_every_smaller_alternative(
    i: int,
    alts: Tuple[Tuple[type, st.SearchStrategy[Any]], ...],
    seed: int,
    force_small: bool
) -> None:
    types: List[type] = [t for t, _ in alts]
    strats: List[st.SearchStrategy[Any]] = [s for _, s in alts]
    combined_strategy: st.SearchStrategy[Any] = st.one_of(*strats)
    if force_small:
        result, value = precisely_shrink(
            combined_strategy,
            is_interesting=lambda x: type(x) is types[i],
            seed=seed
        )
    else:
        result, value = find_random(combined_strategy, lambda x: type(x) is types[i], seed=seed)
    shrunk: List[Tuple[Any, Any]] = shrinks(combined_strategy, result.nodes, allow_sloppy=False, seed=seed * 17)
    shrunk_values: List[Any] = [t for _, t in shrunk]
    for j in range(i):
        assert any(isinstance(x, types[j]) for x in shrunk_values)
