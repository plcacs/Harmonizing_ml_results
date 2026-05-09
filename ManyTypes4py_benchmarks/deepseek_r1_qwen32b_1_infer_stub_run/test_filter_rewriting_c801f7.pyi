import datetime as dt
import decimal
import math
import operator
import re
from fractions import Fraction
from functools import partial
from sys import float_info
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

A_FEW: int = ...

def test_filter_rewriting_ints(
    data: LazyStrategy,
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool],
    start: Optional[int],
    end: Optional[int]
) -> None: ...

def test_filter_rewriting_floats(
    data: LazyStrategy,
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool],
    min_value: Union[float, math.inf],
    max_value: Union[float, math.inf]
) -> None: ...

def test_rewrite_unsatisfiable_filter(
    s: LazyStrategy,
    pred: Callable[[Any], bool]
) -> None: ...

def test_erroring_rewrite_unsatisfiable_filter(
    s: LazyStrategy,
    pred: Callable[[Any], bool]
) -> None: ...

def test_misc_sat_filter_rewrites(
    data: LazyStrategy,
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool]
) -> None: ...

def test_misc_unsat_filter_rewrites(
    data: LazyStrategy,
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool]
) -> None: ...

def test_applying_noop_filter_returns_self(
    strategy: LazyStrategy
) -> None: ...

def test_unhandled_operator(x: int) -> None: ...

def test_rewriting_does_not_compare_decimal_snan() -> None: ...

def test_rewrite_filter_chains_with_some_unhandled(
    data: LazyStrategy,
    predicates: Iterable[Callable[[Any], bool]],
    s: LazyStrategy
) -> None: ...

def test_rewriting_partially_understood_filters(
    data: LazyStrategy,
    start: Optional[int],
    end: Optional[int],
    predicate: Callable[[Any], bool]
) -> None: ...

def test_sequence_filter_rewriting(
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool]
) -> None: ...

def test_warns_on_suspicious_string_methods(
    method: Callable[[str], Any]
) -> None: ...

def test_bumps_min_size_and_filters_for_content_str_methods(
    method: Callable[[str], bool]
) -> None: ...

def test_isidentifier_filter_properly_rewritten(
    al: Optional[str],
    data: LazyStrategy
) -> None: ...

def test_isidentifer_filter_unsatisfiable(al: str) -> None: ...

def test_filter_floats_can_skip_subnormals(
    op: Callable[[float, float], bool],
    attr: str,
    value: float,
    expected: float
) -> None: ...

def test_filter_rewriting_text_partial_len(
    data: LazyStrategy,
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool],
    start: int,
    end: Union[int, math.inf]
) -> None: ...

def test_can_rewrite_multiple_length_filters_if_not_lambdas(
    data: LazyStrategy
) -> None: ...

def test_filter_rewriting_text_lambda_len(
    data: LazyStrategy,
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool],
    start: int,
    end: Union[int, math.inf]
) -> None: ...

def test_filter_rewriting_lambda_len_unique_elements(
    data: LazyStrategy,
    strategy: LazyStrategy,
    predicate: Callable[[Any], bool],
    start: int,
    end: int
) -> None: ...

def test_does_not_rewrite_unsatisfiable_len_filter(
    predicate: Callable[[Any], bool]
) -> None: ...

def test_regex_filter_rewriting(
    data: LazyStrategy,
    strategy: LazyStrategy,
    pattern: Union[str, bytes],
    method: str
) -> None: ...

def test_error_on_method_which_requires_multiple_args() -> None: ...

def test_dates_filter_rewriting() -> None: ...