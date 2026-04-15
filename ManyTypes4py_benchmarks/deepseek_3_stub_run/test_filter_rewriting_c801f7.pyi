import datetime
import decimal
import math
import operator
import re
from fractions import Fraction
from functools import partial
from sys import float_info
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.errors import HypothesisWarning, Unsatisfiable
from hypothesis.strategies._internal.core import DataObject
from hypothesis.strategies._internal.lazy import LazyStrategy
from hypothesis.strategies._internal.numbers import FloatStrategy, IntegersStrategy
from hypothesis.strategies._internal.strategies import FilteredStrategy, MappedStrategy
from hypothesis.strategies._internal.strings import BytesStrategy, TextStrategy
from hypothesis.strategies._internal import SearchStrategy

T = TypeVar("T")
S = TypeVar("S")

A_FEW: int = ...

@pytest.mark.parametrize('strategy, predicate, start, end', ...)
@settings(max_examples=A_FEW)
@given(data=...)
def test_filter_rewriting_ints(
    data: DataObject,
    strategy: SearchStrategy[int],
    predicate: Callable[[int], bool],
    start: Optional[int],
    end: Optional[int],
) -> None: ...

@pytest.mark.parametrize('strategy, predicate, min_value, max_value', ...)
@settings(max_examples=A_FEW)
@given(data=...)
def test_filter_rewriting_floats(
    data: DataObject,
    strategy: SearchStrategy[float],
    predicate: Callable[[float], bool],
    min_value: float,
    max_value: float,
) -> None: ...

@pytest.mark.parametrize('pred', ...)
@pytest.mark.parametrize('s', ...)
def test_rewrite_unsatisfiable_filter(
    s: Union[SearchStrategy[int], SearchStrategy[float]],
    pred: Callable[[Union[int, float]], bool],
) -> None: ...

@pytest.mark.parametrize('pred', ...)
@pytest.mark.parametrize('s', ...)
@fails_with(Unsatisfiable)
def test_erroring_rewrite_unsatisfiable_filter(
    s: Union[SearchStrategy[int], SearchStrategy[float]],
    pred: Callable[[Union[int, float]], bool],
) -> None: ...

@pytest.mark.parametrize('strategy, predicate', ...)
@given(data=...)
def test_misc_sat_filter_rewrites(
    data: DataObject,
    strategy: SearchStrategy[float],
    predicate: Callable[[float], bool],
) -> None: ...

@pytest.mark.parametrize('strategy, predicate', ...)
@given(data=...)
def test_misc_unsat_filter_rewrites(
    data: DataObject,
    strategy: SearchStrategy[float],
    predicate: Callable[[float], bool],
) -> None: ...

@given(st.integers(0, 2).filter(partial(operator.ne, 1)))
def test_unhandled_operator(x: int) -> None: ...

def test_rewriting_does_not_compare_decimal_snan() -> None: ...

@pytest.mark.parametrize('strategy', ...)
def test_applying_noop_filter_returns_self(
    strategy: Union[SearchStrategy[int], SearchStrategy[float]],
) -> None: ...

def mod2(x: int) -> int: ...

Y: int = ...

@pytest.mark.parametrize('s', ...)
@given(data=..., predicates=...)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_rewrite_filter_chains_with_some_unhandled(
    data: DataObject,
    predicates: Sequence[Callable[[Union[int, float]], bool]],
    s: Union[SearchStrategy[int], SearchStrategy[float]],
) -> None: ...

class NotAFunction:
    def __call__(self, bar: Any) -> bool: ...

lambda_without_source: Callable[[Any], Any] = ...

@pytest.mark.parametrize('start, end, predicate', ...)
@given(data=...)
def test_rewriting_partially_understood_filters(
    data: DataObject,
    start: Optional[int],
    end: Optional[int],
    predicate: Callable[[int], bool],
) -> None: ...

@pytest.mark.parametrize('strategy', ...)
@pytest.mark.parametrize('predicate', ...)
def test_sequence_filter_rewriting(
    strategy: SearchStrategy[Any],
    predicate: Callable[[Any], Any],
) -> None: ...

@pytest.mark.parametrize('method', ...)
def test_warns_on_suspicious_string_methods(
    method: Callable[[str], str],
) -> None: ...

@pytest.mark.parametrize('method', ...)
def test_bumps_min_size_and_filters_for_content_str_methods(
    method: Callable[[str], bool],
) -> None: ...

@pytest.mark.parametrize('al', ...)
@given(data())
def test_isidentifier_filter_properly_rewritten(
    al: Optional[str],
    data: DataObject,
) -> None: ...

@pytest.mark.parametrize('al', ...)
def test_isidentifer_filter_unsatisfiable(al: str) -> None: ...

@pytest.mark.parametrize('op, attr, value, expected', ...)
def test_filter_floats_can_skip_subnormals(
    op: Callable[[float, float], bool],
    attr: str,
    value: float,
    expected: float,
) -> None: ...

@pytest.mark.parametrize('strategy, predicate, start, end', ...)
@settings(max_examples=A_FEW)
@given(data=...)
def test_filter_rewriting_text_partial_len(
    data: DataObject,
    strategy: SearchStrategy[str],
    predicate: Callable[[str], bool],
    start: int,
    end: Union[int, float],
) -> None: ...

@given(data=...)
def test_can_rewrite_multiple_length_filters_if_not_lambdas(
    data: DataObject,
) -> None: ...

@pytest.mark.parametrize('predicate, start, end', ...)
@pytest.mark.parametrize('strategy', ...)
@settings(max_examples=A_FEW)
@given(data=...)
def test_filter_rewriting_text_lambda_len(
    data: DataObject,
    strategy: SearchStrategy[Any],
    predicate: Callable[[Any], bool],
    start: int,
    end: Union[int, float],
) -> None: ...

two: int = ...

@pytest.mark.parametrize('predicate, start, end', ...)
@pytest.mark.parametrize('strategy', ...)
@settings(max_examples=A_FEW)
@given(data=...)
def test_filter_rewriting_lambda_len_unique_elements(
    data: DataObject,
    strategy: SearchStrategy[List[int]],
    predicate: Callable[[List[int]], bool],
    start: int,
    end: int,
) -> None: ...

@pytest.mark.parametrize('predicate', ...)
def test_does_not_rewrite_unsatisfiable_len_filter(
    predicate: Callable[[List[None]], bool],
) -> None: ...

@pytest.mark.parametrize('method', ...)
@pytest.mark.parametrize('strategy, pattern', ...)
@settings(max_examples=A_FEW)
@given(data=...)
def test_regex_filter_rewriting(
    data: DataObject,
    strategy: Union[SearchStrategy[str], SearchStrategy[bytes]],
    pattern: Union[str, bytes],
    method: str,
) -> None: ...

@fails_with(TypeError)
@given(st.text().filter(re.compile('abc').sub))
def test_error_on_method_which_requires_multiple_args(_: Any) -> None: ...

def test_dates_filter_rewriting() -> None: ...