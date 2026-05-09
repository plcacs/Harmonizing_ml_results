import datetime as dt
import decimal
import math
import operator
import re
from fractions import Fraction
from typing import Any, Callable, Optional, Union, Type, cast
from hypothesis.strategies import Strategy, SearchStrategy
from hypothesis.strategies._internal.core import Data
from hypothesis.strategies._internal.lazy import LazyStrategy
from hypothesis.strategies._internal.numbers import FloatStrategy, IntegersStrategy
from hypothesis.strategies._internal.strategies import FilteredStrategy, MappedStrategy
from hypothesis.strategies._internal.strings import BytesStrategy, TextStrategy

A_FEW: int ...
Y: int ...
two: int ...
lambda_without_source: Callable[[Any], bool] ...

def test_filter_rewriting_ints(
    data: Data,
    strategy: Strategy[int],
    predicate: Callable[[int], bool],
    start: Optional[int],
    end: Optional[int],
) -> None: ...

def test_filter_rewriting_floats(
    data: Data,
    strategy: Strategy[float],
    predicate: Callable[[float], bool],
    min_value: float,
    max_value: float,
) -> None: ...

def test_rewrite_unsatisfiable_filter(s: Strategy[Any], pred: Callable[[Any], bool]) -> None: ...

def test_erroring_rewrite_unsatisfiable_filter(s: Strategy[Any], pred: Callable[[Any], bool]) -> None: ...

def test_misc_sat_filter_rewrites(
    data: Data,
    strategy: Strategy[float],
    predicate: Callable[[float], bool],
) -> None: ...

def test_misc_unsat_filter_rewrites(
    strategy: Strategy[float],
    predicate: Callable[[float], bool],
) -> None: ...

def test_unhandled_operator(x: int) -> None: ...

def test_rewriting_does_not_compare_decimal_snan() -> None: ...

def test_applying_noop_filter_returns_self(strategy: Strategy[Any]) -> None: ...

def mod2(x: int) -> int: ...

def test_rewrite_filter_chains_with_some_unhandled(
    data: Data,
    predicates: list[Callable[[Any], bool]],
    s: Strategy[Any],
) -> None: ...

class NotAFunction:
    def __call__(self, bar: Any) -> bool: ...

def test_rewriting_partially_understood_filters(
    data: Data,
    start: Optional[int],
    end: Optional[int],
    predicate: Callable[[int], bool],
) -> None: ...

def test_sequence_filter_rewriting(strategy: Strategy[Any], predicate: Callable[[Any], Any]) -> None: ...

def test_warns_on_suspicious_string_methods(method: Callable[[str], str]) -> None: ...

def test_bumps_min_size_and_filters_for_content_str_methods(method: Callable[[str], bool]) -> None: ...

def test_isidentifier_filter_properly_rewritten(al: Optional[str], data: Data) -> None: ...

def test_isidentifer_filter_unsatisfiable(al: str) -> None: ...

def test_filter_floats_can_skip_subnormals(
    op: Callable[[float, float], bool],
    attr: str,
    value: float,
    expected: float,
) -> None: ...

def test_filter_rewriting_text_partial_len(
    data: Data,
    strategy: Strategy[str],
    predicate: Callable[[str], bool],
    start: Union[int, float],
    end: Union[int, float],
) -> None: ...

def test_can_rewrite_multiple_length_filters_if_not_lambdas(data: Data) -> None: ...

def test_filter_rewriting_text_lambda_len(
    data: Data,
    strategy: Strategy[Any],
    predicate: Callable[[Any], bool],
    start: Union[int, float],
    end: Union[int, float],
) -> None: ...

def test_filter_rewriting_lambda_len_unique_elements(
    data: Data,
    strategy: Strategy[Any],
    predicate: Callable[[Any], bool],
    start: int,
    end: int,
) -> None: ...

def test_does_not_rewrite_unsatisfiable_len_filter(predicate: Callable[[Any], bool]) -> None: ...

def test_regex_filter_rewriting(
    data: Data,
    strategy: Strategy[Union[str, bytes]],
    pattern: Union[str, bytes],
    method: str,
) -> None: ...

def test_error_on_method_which_requires_multiple_args(_) -> None: ...

def test_dates_filter_rewriting() -> None: ...