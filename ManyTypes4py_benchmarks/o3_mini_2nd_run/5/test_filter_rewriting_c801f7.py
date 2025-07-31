from __future__ import annotations
import datetime as dt
import decimal
import math
import operator
import re
from fractions import Fraction
from functools import partial
from sys import float_info
from typing import Any, Callable, List, Optional, Tuple, Union

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.errors import HypothesisWarning, Unsatisfiable
from hypothesis.internal.conjecture.providers import COLLECTION_DEFAULT_MAX_SIZE
from hypothesis.internal.filtering import max_len, min_len
from hypothesis.internal.floats import next_down, next_up
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.core import data
from hypothesis.strategies._internal.lazy import LazyStrategy, unwrap_strategies
from hypothesis.strategies._internal.numbers import FloatStrategy, IntegersStrategy
from hypothesis.strategies._internal.strategies import FilteredStrategy, MappedStrategy
from hypothesis.strategies._internal.strings import BytesStrategy, TextStrategy
from tests.common.debug import check_can_generate_examples
from tests.common.utils import fails_with

A_FEW: int = 15


@pytest.mark.parametrize(
    "strategy, predicate, start, end",
    [
        (st.integers(1, 5), math.isfinite, 1, 5),
        (st.integers(1, 5), partial(operator.lt, 3), 4, 5),
        (st.integers(1, 5), partial(operator.le, 3), 3, 5),
        (st.integers(1, 5), partial(operator.eq, 3), 3, 3),
        (st.integers(1, 5), partial(operator.ge, 3), 1, 3),
        (st.integers(1, 5), partial(operator.gt, 3), 1, 2),
        (st.integers(1, 5), partial(operator.lt, 3.5), 4, 5),
        (st.integers(1, 5), partial(operator.le, 3.5), 4, 5),
        (st.integers(1, 5), partial(operator.ge, 3.5), 1, 3),
        (st.integers(1, 5), partial(operator.gt, 3.5), 1, 3),
        (st.integers(1, 5), partial(operator.lt, -math.inf), 1, 5),
        (st.integers(1, 5), partial(operator.gt, math.inf), 1, 5),
        (st.integers(min_value=1), partial(operator.lt, 3), 4, None),
        (st.integers(min_value=1), partial(operator.le, 3), 3, None),
        (st.integers(max_value=5), partial(operator.ge, 3), None, 3),
        (st.integers(max_value=5), partial(operator.gt, 3), None, 2),
        (st.integers(), partial(operator.lt, 3), 4, None),
        (st.integers(), partial(operator.le, 3), 3, None),
        (st.integers(), partial(operator.eq, 3), 3, 3),
        (st.integers(), partial(operator.ge, 3), None, 3),
        (st.integers(), partial(operator.gt, 3), None, 2),
        (st.integers(), lambda x: x < 3, None, 2),
        (st.integers(), lambda x: x <= 3, None, 3),
        (st.integers(), lambda x: x == 3, 3, 3),
        (st.integers(), lambda x: x >= 3, 3, None),
        (st.integers(), lambda x: x > 3, 4, None),
        (st.integers(), lambda x: 3 > x, None, 2),
        (st.integers(), lambda x: 3 >= x, None, 3),
        (st.integers(), lambda x: 3 == x, 3, 3),
        (st.integers(), lambda x: 3 <= x, 3, None),
        (st.integers(), lambda x: 3 < x, 4, None),
        (st.integers(), lambda x: 0 < x < 5, 1, 4),
        (st.integers(), lambda x: 0 < x >= 1, 1, None),
        (st.integers(), lambda x: 1 > x <= 0, None, 0),
        (st.integers(), lambda x: x > 0 and x > 0, 1, None),
        (st.integers(), lambda x: x < 1 and x < 1, None, 0),
        (st.integers(), lambda x: x > 1 and x > 0, 2, None),
        (st.integers(), lambda x: x < 1 and x < 2, None, 0),
    ],
    ids=get_pretty_function_description,
)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_filter_rewriting_ints(
    data: Any,
    strategy: st.SearchStrategy[int],
    predicate: Callable[[int], bool],
    start: Optional[int],
    end: Optional[int],
) -> None:
    s: LazyStrategy = strategy.filter(predicate)
    assert isinstance(s, LazyStrategy)
    assert isinstance(s.wrapped_strategy, IntegersStrategy)
    assert s.wrapped_strategy.start == start
    assert s.wrapped_strategy.end == end
    value: int = data.draw(s)
    assert predicate(value)


@pytest.mark.parametrize(
    "strategy, predicate, min_value, max_value",
    [
        (st.floats(1, 5), partial(operator.lt, 3), next_up(3.0), 5),
        (st.floats(1, 5), partial(operator.le, 3), 3, 5),
        (st.floats(1, 5), partial(operator.eq, 3), 3, 3),
        (st.floats(1, 5), partial(operator.ge, 3), 1, 3),
        (st.floats(1, 5), partial(operator.gt, 3), 1, next_down(3.0)),
        (st.floats(1, 5), partial(operator.lt, 3.5), next_up(3.5), 5),
        (st.floats(1, 5), partial(operator.le, 3.5), 3.5, 5),
        (st.floats(1, 5), partial(operator.ge, 3.5), 1, 3.5),
        (st.floats(1, 5), partial(operator.gt, 3.5), 1, next_down(3.5)),
        (st.floats(1, 5), partial(operator.lt, -math.inf), 1, 5),
        (st.floats(1, 5), partial(operator.gt, math.inf), 1, 5),
        (st.floats(min_value=1), partial(operator.lt, 3), next_up(3.0), math.inf),
        (st.floats(min_value=1), partial(operator.le, 3), 3, math.inf),
        (st.floats(max_value=5), partial(operator.ge, 3), -math.inf, 3),
        (st.floats(max_value=5), partial(operator.gt, 3), -math.inf, next_down(3.0)),
        (st.floats(), partial(operator.lt, 3), next_up(3.0), math.inf),
        (st.floats(), partial(operator.le, 3), 3, math.inf),
        (st.floats(), partial(operator.eq, 3), 3, 3),
        (st.floats(), partial(operator.ge, 3), -math.inf, 3),
        (st.floats(), partial(operator.gt, 3), -math.inf, next_down(3.0)),
        (st.floats(), lambda x: x < 3, -math.inf, next_down(3.0)),
        (st.floats(), lambda x: x <= 3, -math.inf, 3),
        (st.floats(), lambda x: x == 3, 3, 3),
        (st.floats(), lambda x: x >= 3, 3, math.inf),
        (st.floats(), lambda x: x > 3, next_up(3.0), math.inf),
        (st.floats(), lambda x: 3 > x, -math.inf, next_down(3.0)),
        (st.floats(), lambda x: 3 >= x, -math.inf, 3),
        (st.floats(), lambda x: 3 == x, 3, 3),
        (st.floats(), lambda x: 3 <= x, 3, math.inf),
        (st.floats(), lambda x: 3 < x, next_up(3.0), math.inf),
        (st.floats(), lambda x: 0 < x < 5, next_up(0.0), next_down(5.0)),
        (st.floats(), lambda x: 0 < x >= 1, 1, math.inf),
        (st.floats(), lambda x: 1 > x <= 0, -math.inf, 0),
        (st.floats(), lambda x: x > 0 and x > 0, next_up(0.0), math.inf),
        (st.floats(), lambda x: x < 1 and x < 1, -math.inf, next_down(1.0)),
        (st.floats(), lambda x: x > 1 and x > 0, next_up(1.0), math.inf),
        (st.floats(), lambda x: x < 1 and x < 2, -math.inf, next_down(1.0)),
        (st.floats(), math.isfinite, next_up(-math.inf), next_down(math.inf)),
    ],
    ids=get_pretty_function_description,
)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_filter_rewriting_floats(
    data: Any,
    strategy: st.SearchStrategy[float],
    predicate: Callable[[float], bool],
    min_value: float,
    max_value: float,
) -> None:
    s: LazyStrategy = strategy.filter(predicate)
    assert isinstance(s, LazyStrategy)
    assert isinstance(s.wrapped_strategy, FloatStrategy)
    assert s.wrapped_strategy.min_value == min_value
    assert s.wrapped_strategy.max_value == max_value
    value: float = data.draw(s)
    assert predicate(value)


@pytest.mark.parametrize(
    "pred", [
        math.isinf,
        math.isnan,
        partial(operator.lt, 6),
        partial(operator.eq, Fraction(10, 3)),
        partial(operator.ge, 0),
        partial(operator.lt, math.inf),
        partial(operator.gt, -math.inf)
    ]
)
@pytest.mark.parametrize(
    "s", [st.integers(1, 5), st.floats(1, 5)]
)
def test_rewrite_unsatisfiable_filter(
    s: st.SearchStrategy[Any],
    pred: Callable[[Any], bool],
) -> None:
    assert s.filter(pred).is_empty


@pytest.mark.parametrize(
    "pred", [partial(operator.eq, 'numbers are never equal to strings')]
)
@pytest.mark.parametrize(
    "s", [st.integers(1, 5), st.floats(1, 5)]
)
@fails_with(Unsatisfiable)
def test_erroring_rewrite_unsatisfiable_filter(
    s: st.SearchStrategy[Any],
    pred: Callable[[Any], bool],
) -> None:
    check_can_generate_examples(s.filter(pred))


@pytest.mark.parametrize(
    "strategy, predicate", [
        (st.floats(), math.isinf),
        (st.floats(0, math.inf), math.isinf),
        (st.floats(), math.isnan)
    ]
)
@given(data=st.data())
def test_misc_sat_filter_rewrites(
    data: Any,
    strategy: st.SearchStrategy[float],
    predicate: Callable[[float], bool],
) -> None:
    s = strategy.filter(predicate).wrapped_strategy
    assert not isinstance(s, FloatStrategy)
    value: float = data.draw(s)
    assert predicate(value)


@pytest.mark.parametrize(
    "strategy, predicate", [
        (st.floats(allow_infinity=False), math.isinf),
        (st.floats(0, math.inf), math.isnan),
        (st.floats(allow_nan=False), math.isnan)
    ]
)
@given(data=st.data())
def test_misc_unsat_filter_rewrites(
    data: Any,
    strategy: st.SearchStrategy[float],
    predicate: Callable[[float], bool],
) -> None:
    assert strategy.filter(predicate).is_empty


@given(st.integers(0, 2).filter(partial(operator.ne, 1)))
def test_unhandled_operator(x: int) -> None:
    assert x in (0, 2)


def test_rewriting_does_not_compare_decimal_snan() -> None:
    s: LazyStrategy = st.integers(1, 5).filter(partial(operator.eq, decimal.Decimal("snan")))
    s.wrapped_strategy  # type: ignore
    with pytest.raises(decimal.InvalidOperation):
        check_can_generate_examples(s)


@pytest.mark.parametrize("strategy", [st.integers(0, 1), st.floats(0, 1)], ids=repr)
def test_applying_noop_filter_returns_self(strategy: st.SearchStrategy[Any]) -> None:
    s = strategy.wrapped_strategy
    s2 = s.filter(partial(operator.le, -1)).filter(partial(operator.ge, 2))
    assert s is s2


def mod2(x: int) -> int:
    return x % 2


Y: int = 2 ** 20


@pytest.mark.parametrize(
    "s",
    [st.integers(1, 5), st.floats(1, 5)],
)
@given(data=st.data(), predicates=st.permutations([
    partial(operator.lt, 1),
    partial(operator.le, 2),
    partial(operator.ge, 4),
    partial(operator.gt, 5),
    mod2,
    lambda x: x > 2 or x % 7,
    lambda x: 0 < x <= Y
]),)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_rewrite_filter_chains_with_some_unhandled(
    data: Any,
    predicates: List[Callable[[Any], bool]],
    s: st.SearchStrategy[Any],
) -> None:
    for p in predicates:
        s = s.filter(p)
    value = data.draw(s)
    for p in predicates:
        assert p(value), f"p={p!r}, value={value}"
    unwrapped = s.wrapped_strategy
    assert isinstance(unwrapped, FilteredStrategy)
    assert isinstance(unwrapped.filtered_strategy, (IntegersStrategy, FloatStrategy))
    for pred in unwrapped.flat_conditions:
        assert pred is mod2 or pred.__name__ == "<lambda>"


class NotAFunction:
    def __call__(self, bar: Any) -> bool:
        return True


lambda_without_source: Callable[[Any], Any] = eval("lambda x: x > 2", {}, {})
assert get_pretty_function_description(lambda_without_source) == "lambda x: <unknown>"


@pytest.mark.parametrize(
    "start, end, predicate",
    [
        (1, 4, lambda x: 0 < x < 5 and x % 7),
        (0, 9, lambda x: 0 <= x < 10 and x % 3),
        (1, None, lambda x: 0 < x <= Y),
        (None, None, lambda x: x == x),
        (None, None, lambda x: 1 == 1),
        (None, None, lambda x: 1 <= 2),
        (None, None, lambda x: x != 0),
        (None, None, NotAFunction()),
        (None, None, lambda_without_source),
        (None, None, lambda x, y=2: x >= 0),
    ],
)
@given(data=st.data())
def test_rewriting_partially_understood_filters(
    data: Any,
    start: Optional[int],
    end: Optional[int],
    predicate: Callable[..., bool],
) -> None:
    s: FilteredStrategy = st.integers().filter(predicate).wrapped_strategy  # type: ignore
    assert isinstance(s, FilteredStrategy)
    assert isinstance(s.filtered_strategy, IntegersStrategy)
    assert s.filtered_strategy.start == start
    assert s.filtered_strategy.end == end
    assert s.flat_conditions == (predicate,)
    value: int = data.draw(s)
    assert predicate(value)


@pytest.mark.parametrize(
    "strategy, predicate",
    [
        (st.text(), bool),
        (st.text(min_size=2), len),
        (st.lists(st.none()), tuple),
        (st.lists(st.none(), min_size=2), list),
        (st.text(), lambda x: x),
    ],
    ids=get_pretty_function_description,
)
def test_sequence_filter_rewriting(
    strategy: st.SearchStrategy[Any],
    predicate: Callable[[Any], Any],
) -> None:
    s = unwrap_strategies(strategy)
    fs = s.filter(predicate)
    assert not isinstance(fs, FilteredStrategy)
    if s.min_size > 0:
        assert fs is s
    else:
        assert fs.min_size == 1


@pytest.mark.parametrize("method", [str.lower, str.title, str.upper])
def test_warns_on_suspicious_string_methods(method: Callable[[str], str]) -> None:
    s = unwrap_strategies(st.text())
    with pytest.warns(HypothesisWarning, match=r"this allows all nonempty strings!  Did you mean"):
        fs = s.filter(method)
    assert fs.min_size == 1


@pytest.mark.parametrize("method", [str.isalnum])
def test_bumps_min_size_and_filters_for_content_str_methods(method: Callable[[str], bool]) -> None:
    s = unwrap_strategies(st.text())
    fs = s.filter(method)
    assert fs.filtered_strategy.min_size == 1
    assert fs.flat_conditions == (method,)


@pytest.mark.parametrize("al", [None, "cdef123", "cd12¥¦§©"])
@given(data=st.data())
def test_isidentifier_filter_properly_rewritten(
    al: Optional[str],
    data: Any,
) -> None:
    if al is None:
        example = data.draw(st.text().filter(str.isidentifier))
    else:
        example = data.draw(st.text(alphabet=al).filter(str.isidentifier))
        assert set(example).issubset(set(al))
    assert example.isidentifier()


@pytest.mark.parametrize("al", ["¥¦§©"])
def test_isidentifer_filter_unsatisfiable(al: str) -> None:
    fs = st.text(alphabet=al).filter(str.isidentifier)
    with pytest.raises(Unsatisfiable):
        check_can_generate_examples(fs)


@pytest.mark.parametrize(
    "op, attr, value, expected",
    [
        (operator.lt, "min_value", -float_info.min / 2, 0),
        (operator.lt, "min_value", float_info.min / 2, float_info.min),
        (operator.gt, "max_value", float_info.min / 2, 0),
        (operator.gt, "max_value", -float_info.min / 2, -float_info.min),
    ],
)
def test_filter_floats_can_skip_subnormals(
    op: Callable[[float, float], bool],
    attr: str,
    value: float,
    expected: float,
) -> None:
    base = st.floats(allow_subnormal=False).filter(partial(op, value))
    assert getattr(base.wrapped_strategy, attr) == expected


@pytest.mark.parametrize(
    "strategy, predicate, start, end",
    [
        (st.text(min_size=1, max_size=5), partial(min_len, 3), 3, 5),
        (st.text(min_size=1, max_size=5), partial(max_len, 3), 1, 3),
        (st.text(min_size=1), partial(min_len, 3), 3, math.inf),
        (st.text(min_size=1), partial(max_len, 3), 1, 3),
        (st.text(max_size=5), partial(min_len, 3), 3, 5),
        (st.text(max_size=5), partial(max_len, 3), 0, 3),
        (st.text(), partial(min_len, 3), 3, math.inf),
        (st.text(), partial(max_len, 3), 0, 3),
    ],
    ids=get_pretty_function_description,
)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_filter_rewriting_text_partial_len(
    data: Any,
    strategy: st.SearchStrategy[str],
    predicate: Callable[[str], bool],
    start: Union[int, float],
    end: Union[int, float],
) -> None:
    s: LazyStrategy = strategy.filter(predicate)
    inner = unwrap_strategies(s)
    assert isinstance(inner, TextStrategy)
    assert inner.min_size == start
    assert inner.max_size == end
    value: str = data.draw(s)
    assert predicate(value)


@given(data=st.data())
def test_can_rewrite_multiple_length_filters_if_not_lambdas(data: Any) -> None:
    s: st.SearchStrategy[str] = st.text(min_size=1, max_size=5).filter(partial(min_len, 2)).filter(partial(max_len, 4))
    assert isinstance(s, LazyStrategy)
    inner = unwrap_strategies(s)
    assert isinstance(inner, TextStrategy)
    assert inner.min_size == 2
    assert inner.max_size == 4
    value: str = data.draw(s)
    assert 2 <= len(value) <= 4


@pytest.mark.parametrize(
    "predicate, start, end",
    [
        (lambda x: len(x) < 3, 0, 2),
        (lambda x: len(x) <= 3, 0, 3),
        (lambda x: len(x) == 3, 3, 3),
        (lambda x: len(x) >= 3, 3, math.inf),
        (lambda x: len(x) > 3, 4, math.inf),
        (lambda x: 3 > len(x), 0, 2),
        (lambda x: 3 >= len(x), 0, 3),
        (lambda x: 3 == len(x), 3, 3),
        (lambda x: 3 <= len(x), 3, math.inf),
        (lambda x: 3 < len(x), 4, math.inf),
        (lambda x: 0 < len(x) < 5, 1, 4),
        (lambda x: 0 < len(x) >= 1, 1, math.inf),
        (lambda x: 1 > len(x) <= 0, 0, 0),
        (lambda x: len(x) > 0 and len(x) > 0, 1, math.inf),
        (lambda x: len(x) < 1 and len(x) < 1, 0, 0),
        (lambda x: len(x) > 1 and len(x) > 0, 2, math.inf),
        (lambda x: len(x) < 1 and len(x) < 2, 0, 0),
    ],
    ids=get_pretty_function_description,
)
@pytest.mark.parametrize(
    "strategy",
    [
        st.text(),
        st.lists(st.integers()),
        st.lists(st.integers(), unique=True),
        st.lists(st.sampled_from([1, 2, 3])),
        st.binary(),
        st.sets(st.integers()),
        st.frozensets(st.integers()),
        st.dictionaries(st.integers(), st.none()),
        st.lists(st.integers(), unique_by=lambda x: x % 17).map(tuple),
    ],
    ids=get_pretty_function_description,
)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_filter_rewriting_text_lambda_len(
    data: Any,
    strategy: st.SearchStrategy[Any],
    predicate: Callable[[Any], bool],
    start: Union[int, float],
    end: Union[int, float],
) -> None:
    s = strategy.filter(predicate)
    unwrapped_nofilter = unwrap_strategies(strategy)
    unwrapped = unwrap_strategies(s)
    was_mapped: bool = isinstance(unwrapped, MappedStrategy)
    if was_mapped:
        unwrapped = unwrapped.mapped_strategy
    assert isinstance(unwrapped, FilteredStrategy), f"unwrapped={unwrapped!r} type(unwrapped)={type(unwrapped)!r}"
    assert isinstance(unwrapped.filtered_strategy, type(unwrapped_nofilter.mapped_strategy if was_mapped else unwrapped_nofilter))
    for pred in unwrapped.flat_conditions:
        assert pred.__name__ == "<lambda>"
    if isinstance(unwrapped.filtered_strategy, MappedStrategy):
        unwrapped = unwrapped.filtered_strategy.mapped_strategy
    if isinstance(unwrapped_nofilter, BytesStrategy) and end == math.inf:
        end = COLLECTION_DEFAULT_MAX_SIZE
    assert unwrapped.filtered_strategy.min_size == start
    assert unwrapped.filtered_strategy.max_size == end
    value = data.draw(s)
    assert predicate(value)


two: int = 2


@pytest.mark.parametrize(
    "predicate, start, end",
    [
        (lambda x: len(x) < 3, 0, 2),
        (lambda x: len(x) <= 3, 0, 3),
        (lambda x: len(x) == 3, 3, 3),
        (lambda x: len(x) >= 3, 3, 3),
        (lambda x: 3 > len(x), 0, 2),
        (lambda x: 3 >= len(x), 0, 3),
        (lambda x: 3 == len(x), 3, 3),
        (lambda x: 3 <= len(x), 3, 3),
        (lambda x: 0 < len(x) < 5, 1, 3),
        (lambda x: 0 < len(x) >= 1, 1, 3),
        (lambda x: 1 > len(x) <= 0, 0, 0),
        (lambda x: len(x) > 0 and len(x) > 0, 1, 3),
        (lambda x: len(x) < 1 and len(x) < 1, 0, 0),
        (lambda x: len(x) > 1 and len(x) > 0, 2, 3),
        (lambda x: len(x) < 1 and len(x) < 2, 0, 0),
        (lambda x: 1 <= len(x) <= two, 1, 3),
        (lambda x: two <= len(x) <= 4, 0, 3),
    ],
    ids=get_pretty_function_description,
)
@pytest.mark.parametrize(
    "strategy",
    [st.lists(st.sampled_from([1, 2, 3]), unique=True)],
    ids=get_pretty_function_description,
)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_filter_rewriting_lambda_len_unique_elements(
    data: Any,
    strategy: st.SearchStrategy[Any],
    predicate: Callable[[Any], bool],
    start: Union[int, float],
    end: Union[int, float],
) -> None:
    s = strategy.filter(predicate)
    unwrapped = unwrap_strategies(s)
    assert isinstance(unwrapped, FilteredStrategy)
    assert isinstance(unwrapped.filtered_strategy, type(unwrap_strategies(strategy)))
    for pred in unwrapped.flat_conditions:
        assert pred.__name__ == "<lambda>"
    assert unwrapped.filtered_strategy.min_size == start
    assert unwrapped.filtered_strategy.max_size == end
    value = data.draw(s)
    assert predicate(value)


@pytest.mark.parametrize(
    "predicate", [lambda x: len(x) < 3, lambda x: len(x) > 5],
    ids=get_pretty_function_description,
)
def test_does_not_rewrite_unsatisfiable_len_filter(
    predicate: Callable[[Any], bool],
) -> None:
    strategy = st.lists(st.none(), min_size=4, max_size=4).filter(predicate)
    with pytest.raises(Unsatisfiable):
        check_can_generate_examples(strategy)
    assert not strategy.is_empty


@pytest.mark.parametrize("method", ["match", "search", "findall", "fullmatch", "finditer", "split"])
@pytest.mark.parametrize(
    "strategy, pattern",
    [
        (st.text(), "ab+c"),
        (st.text(), "a|b"),
        (st.text(alphabet="abcdef"), "ab+c"),
        (st.text(min_size=5, max_size=10), "ab+c"),
        (st.binary(), b"ab+c"),
        (st.binary(), b"a|b"),
        (st.binary(min_size=5, max_size=10), b"ab+c"),
    ],
    ids=repr,
)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_regex_filter_rewriting(
    data: Any,
    strategy: st.SearchStrategy[Any],
    pattern: Union[str, bytes],
    method: str,
) -> None:
    predicate = getattr(re.compile(pattern), method)
    s = strategy.filter(predicate)
    if method in ("finditer", "split"):
        msg = r"You applied re.compile\(.+?\).\w+ as a filter, but this allows"
        with pytest.warns(HypothesisWarning, match=msg):
            value = data.draw(s)
    else:
        value = data.draw(s)
    assert predicate(value)


@fails_with(TypeError)
@given(st.text().filter(re.compile("abc").sub))
def test_error_on_method_which_requires_multiple_args(_: Any) -> None:
    pass


def test_dates_filter_rewriting() -> None:
    today: dt.date = dt.date.today()
    assert st.dates().filter(partial(operator.lt, dt.date.max)).is_empty
    assert st.dates().filter(partial(operator.gt, dt.date.min)).is_empty
    assert st.dates(min_value=today).filter(partial(operator.gt, today)).is_empty
    assert st.dates(max_value=today).filter(partial(operator.lt, today)).is_empty
    bare = unwrap_strategies(st.dates())
    assert bare.filter(partial(operator.ge, dt.date.max)) is bare
    assert bare.filter(partial(operator.le, dt.date.min)) is bare
    new = bare.filter(partial(operator.le, today))
    assert not new.is_empty
    assert new is not bare
