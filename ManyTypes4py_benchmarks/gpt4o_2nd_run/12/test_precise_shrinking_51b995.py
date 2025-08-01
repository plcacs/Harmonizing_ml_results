import itertools
from functools import lru_cache
from random import Random
from typing import Callable, List, Tuple, TypeVar, Any
import pytest
from hypothesis import find, settings, strategies as st
from hypothesis.control import BuildContext
from hypothesis.errors import StopTest, UnsatisfiedAssumption
from hypothesis.internal.conjecture.data import ConjectureData, ConjectureResult, Status
from hypothesis.internal.conjecture.engine import ConjectureRunner, ExitReason, RunIsComplete
from hypothesis.internal.conjecture.shrinker import sort_key

T = TypeVar('T')

def safe_draw(data: ConjectureData, strategy: st.SearchStrategy[T]) -> T:
    """Set up just enough of the Hypothesis machinery to use draw on
    a strategy."""
    with BuildContext(data):
        try:
            return data.draw(strategy)
        except UnsatisfiedAssumption:
            data.mark_invalid()

def precisely_shrink(
    strategy: st.SearchStrategy[T],
    is_interesting: Callable[[T], bool] = lambda x: True,
    initial_condition: Callable[[T], bool] = lambda x: True,
    end_marker: st.SearchStrategy[Any] = st.integers(),
    seed: int = 0
) -> Tuple[ConjectureResult, T]:
    """Generates a random value from the strategy and then precisely shrinks it,
    by shrinking it with some value immediately afterwards that is not allowed to
    be modified during shrinking."""
    random = Random(seed)
    while True:
        data = ConjectureData(random=random)
        try:
            initial_value = safe_draw(data, strategy)
        except StopTest:
            continue
        if is_interesting(initial_value) and initial_condition(initial_value):
            break
    target_check_value = safe_draw(data, end_marker)
    initial_choices = data.choices
    replay = ConjectureData.for_choices(initial_choices)
    assert safe_draw(replay, strategy) == initial_value
    assert safe_draw(replay, end_marker) == target_check_value

    def test_function(data: ConjectureData) -> None:
        value = safe_draw(data, strategy)
        check_value = safe_draw(data, end_marker)
        if is_interesting(value) and check_value == target_check_value:
            data.mark_interesting()
    runner = ConjectureRunner(test_function, random=random)
    try:
        buf = runner.cached_test_function_ir(initial_choices)
        assert buf.status == Status.INTERESTING
        assert buf.choices == initial_choices
        assert runner.interesting_examples
        runner.shrink_interesting_examples()
    except RunIsComplete:
        assert runner.exit_reason in (ExitReason.finished, ExitReason.max_shrinks)
    result, = runner.interesting_examples.values()
    data = ConjectureData.for_choices(result.choices)
    result_value = safe_draw(data, strategy)
    data.freeze()
    return (data.as_result(), result_value)

common_strategies_with_types: List[Tuple[type, st.SearchStrategy[Any]]] = [
    (type(None), st.none()),
    (bool, st.booleans()),
    (bytes, st.binary()),
    (str, st.text()),
    (int, st.integers())
]
common_strategies: List[st.SearchStrategy[Any]] = [v for _, v in common_strategies_with_types]

@lru_cache
def minimal_for_strategy(s: st.SearchStrategy[T]) -> Tuple[ConjectureResult, T]:
    return precisely_shrink(s, end_marker=st.none())

def minimal_nodes_for_strategy(s: st.SearchStrategy[T]) -> List[Any]:
    return minimal_for_strategy(s)[0].nodes

def test_strategy_list_is_in_sorted_order() -> None:
    assert common_strategies == sorted(common_strategies, key=lambda s: sort_key(minimal_nodes_for_strategy(s)))

@pytest.mark.parametrize('typ,strat', common_strategies_with_types)
@pytest.mark.parametrize('require_truthy', [False, True])
def test_can_precisely_shrink_values(typ: type, strat: st.SearchStrategy[T], require_truthy: bool) -> None:
    if typ is type(None) and require_truthy:
        pytest.skip('None is falsey')
    if require_truthy:
        cond = bool
    else:
        cond = lambda x: True
    result, shrunk = precisely_shrink(strat, is_interesting=cond)
    assert shrunk == find(strat, cond)

alternatives: List[Tuple[Tuple[type, st.SearchStrategy[Any]], ...]] = [
    comb for n in (2, 3, 4) for comb in itertools.combinations(common_strategies_with_types, n)
]
indexed_alternatives: List[Tuple[int, int, Tuple[Tuple[type, st.SearchStrategy[Any]], ...]]] = [
    (i, j, a) for a in alternatives for i, j in itertools.combinations(range(len(a)), 2)
]

@pytest.mark.parametrize('i,j,a', indexed_alternatives)
@pytest.mark.parametrize('seed', [0, 4389048901])
def test_can_precisely_shrink_alternatives(i: int, j: int, a: Tuple[Tuple[type, st.SearchStrategy[Any]], ...], seed: int) -> None:
    types = [u for u, _ in a]
    combined_strategy = st.one_of(*[v for _, v in a])
    result, value = precisely_shrink(
        combined_strategy,
        initial_condition=lambda x: isinstance(x, types[j]),
        is_interesting=lambda x: not any((isinstance(x, types[k]) for k in range(i))),
        seed=seed
    )
    assert isinstance(value, types[i])

@pytest.mark.parametrize('a', list(itertools.combinations(common_strategies_with_types, 3)))
@pytest.mark.parametrize('seed', [0, 4389048901])
def test_precise_shrink_with_blocker(a: Tuple[Tuple[type, st.SearchStrategy[Any]], ...], seed: int) -> None:
    x, y, z = a
    a = (x, z, y)
    types = [u for u, _ in a]
    combined_strategy = st.one_of(*[v for _, v in a])
    result, value = precisely_shrink(
        combined_strategy,
        initial_condition=lambda x: isinstance(x, types[2]),
        is_interesting=lambda x: True,
        seed=seed
    )
    assert isinstance(value, types[0])

def find_random(s: st.SearchStrategy[T], condition: Callable[[T], bool], seed: int = None) -> Tuple[ConjectureResult, T]:
    random = Random(seed)
    while True:
        data = ConjectureData(random=random)
        try:
            with BuildContext(data=data):
                value = data.draw(s)
                if condition(value):
                    data.freeze()
                    return (data.as_result(), value)
        except (StopTest, UnsatisfiedAssumption):
            continue

def shrinks(
    strategy: st.SearchStrategy[T],
    nodes: List[Any],
    *,
    allow_sloppy: bool = True,
    seed: int = 0
) -> List[Tuple[List[Any], T]]:
    results: dict = {}
    random = Random(seed)
    choices = tuple((n.value for n in nodes))
    if allow_sloppy:

        def test_function(data: ConjectureData) -> None:
            value = safe_draw(data, strategy)
            results[data.nodes] = value
        runner = ConjectureRunner(test_function, settings=settings(max_examples=10 ** 9))
        initial = runner.cached_test_function_ir(choices)
        assert isinstance(initial, ConjectureResult)
        try:
            runner.shrink(initial, lambda x: x.choices == initial.choices)
        except RunIsComplete:
            assert runner.exit_reason in (ExitReason.finished, ExitReason.max_shrinks)
    else:
        trial = ConjectureData(prefix=choices, random=random)
        with BuildContext(trial):
            trial.draw(strategy)
            assert trial.choices == choices, 'choice sequence is already sloppy'
            padding = safe_draw(trial, st.integers())
        initial_choices = trial.choices

        def test_function(data: ConjectureData) -> None:
            value = safe_draw(data, strategy)
            key = data.nodes
            padding_check = safe_draw(data, st.integers())
            if padding_check == padding:
                results[key] = value
        runner = ConjectureRunner(test_function, settings=settings(max_examples=10 ** 9))
        initial = runner.cached_test_function_ir(initial_choices)
        assert len(results) == 1
        try:
            runner.shrink(initial, lambda x: x.choices == initial_choices)
        except RunIsComplete:
            assert runner.exit_reason in (ExitReason.finished, ExitReason.max_shrinks)
    results.pop(nodes)
    seen = set()
    result_list: List[Tuple[List[Any], T]] = []
    for k, v in sorted(results.items(), key=lambda x: sort_key(x[0])):
        t = repr(v)
        if t in seen:
            continue
        seen.add(t)
        result_list.append((k, v))
    return result_list

@pytest.mark.parametrize('a', list(itertools.product(*[common_strategies[1:]] * 2)))
@pytest.mark.parametrize('block_falsey', [False, True])
@pytest.mark.parametrize('allow_sloppy', [False, True])
@pytest.mark.parametrize('seed', [0, 2452, 99085240570])
def test_always_shrinks_to_none(a: Tuple[st.SearchStrategy[Any], ...], seed: int, block_falsey: bool, allow_sloppy: bool) -> None:
    combined_strategy = st.one_of(st.none(), *a)
    result, value = find_random(combined_strategy, lambda x: x is not None)
    shrunk_values = shrinks(combined_strategy, result.nodes, allow_sloppy=allow_sloppy, seed=seed)
    assert shrunk_values[0][1] is None

@pytest.mark.parametrize('i,alts', [(i, alt) for alt in alternatives for i in range(1, len(alt))])
@pytest.mark.parametrize('force_small', [False, True])
@pytest.mark.parametrize('seed', [0, 2452, 99085240570])
def test_can_shrink_to_every_smaller_alternative(i: int, alts: Tuple[Tuple[type, st.SearchStrategy[Any]], ...], seed: int, force_small: bool) -> None:
    types = [t for t, _ in alts]
    strats = [s for _, s in alts]
    combined_strategy = st.one_of(*strats)
    if force_small:
        result, value = precisely_shrink(combined_strategy, is_interesting=lambda x: type(x) is types[i], seed=seed)
    else:
        result, value = find_random(combined_strategy, lambda x: type(x) is types[i], seed=seed)
    shrunk = shrinks(combined_strategy, result.nodes, allow_sloppy=False, seed=seed * 17)
    shrunk_values = [t for _, t in shrunk]
    for j in range(i):
        assert any((isinstance(x, types[j]) for x in shrunk_values))
