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

A_FEW: int = 15

@pytest.mark.parametrize('strategy, predicate, start, end', [(st.integers(1, 5), math.isfinite, 1, 5), (st.integers(1, 5), partial(operator.lt, 3), 4, 5), (st.integers(1, 5), partial(operator.le, 3), 3, 5), (st.integers(1, 5), partial(operator.eq, 3), 3, 3), (st.integers(1, 5), partial(operator.ge, 3), 1, 3), (st.integers(1, 5), partial(operator.gt, 3), 1, 2), (st.integers(1, 5), partial(operator.lt, 3.5), 4, 5), (st.integers(1, 5), partial(operator.le, 3.5), 4, 5), (st.integers(1, 5), partial(operator.ge, 3.5), 1, 3), (st.integers(1, 5), partial(operator.gt, 3.5), 1, 3), (st.integers(min_value=1), partial(operator.lt, 3), 4, None), (st.integers(min_value=1), partial(operator.le, 3), 3, None), (st.integers(max_value=5), partial(operator.ge, 3), None, 3), (st.integers(max_value=5), partial(operator.gt, 3), None, 2), (st.integers(), partial(operator.lt, 3), 4, None), (st.integers(), partial(operator.le, 3), 3, None), (st.integers(), partial(operator.eq, 3), 3, 3), (st.integers(), partial(operator.ge, 3), None, 3), (st.integers(), partial(operator.gt, 3), None, 2), (st.integers(), lambda x: x < 3, None, 2), (st.integers(), lambda x: x <= 3, None, 3), (st.integers(), lambda x: x == 3, 3, 3), (st.integers(), lambda x: x >= 3, 3, None), (st.integers(), lambda x: x > 3, 4, None), (st.integers(), lambda x: 3 > x, None, 2), (st.integers(), lambda x: 3 >= x, None, 3), (st.integers(), lambda x: 3 == x, 3, 3), (st.integers(), lambda x: 3 <= x, 3, None), (st.integers(), lambda x: 3 < x, 4, None), (st.integers(), lambda x: 0 < x < 5, 1, 4), (st.integers(), lambda x: 0 < x >= 1, 1, None), (st.integers(), lambda x: 1 > x <= 0, None, 0), (st.integers(), lambda x: x > 0 and x > 0, 1, None), (st.integers(), lambda x: x < 1 and x < 1, None, 0), (st.integers(), lambda x: x > 1 and x > 0, 2, None), (st.integers(), lambda x: x < 1 and x < 2, None, 0)], ids=get_pretty_function_description)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_filter_rewriting_ints(data, strategy, predicate, start, end):
    s: LazyStrategy[int] = strategy.filter(predicate)
    assert isinstance(s, LazyStrategy)
    assert isinstance(s.wrapped_strategy, IntegersStrategy)
    assert s.wrapped_strategy.start == start
    assert s.wrapped_strategy.end == end
    value: int = data.draw(s)
    assert predicate(value)

@pytest.mark.parametrize('strategy, predicate, min_value, max_value', [(st.floats(1, 5), partial(operator.lt, 3), next_up(3.0), 5), (st.floats(1, 5), partial(operator.le, 3), 3, 5), (st.floats(1, 5), partial(operator.eq, 3), 3, 3), (st.floats(1, 5), partial(operator.ge, 3), 1, 3), (st.floats(1, 5), partial(operator.gt, 3), 1, next_down(3.0)), (st.floats(1, 5), partial(operator.lt, 3.5), next_up(3.5), 5), (st.floats(1, 5), partial(operator.le, 3.5), 3.5, 5), (st.floats(1, 5), partial(operator.ge, 3.5), 1, 3.5), (st.floats(1, 5), partial(operator.gt, 3.5), 1, next_down(3.5)), (st.floats(1, 5), partial(operator.lt, -math.inf), 1, 5), (st.floats(1, 5), partial(operator.gt, math.inf), 1, 5), (st.floats(min_value=1), partial(operator.lt, 3), next_up(3.0), math.inf), (st.floats(min_value=1), partial(operator.le, 3), 3, math.inf), (st.floats(max_value=5), partial(operator.ge, 3), -math.inf, 3), (st.floats(max_value=5), partial(operator.gt, 3), -math.inf, next_down(3.0)), (st.floats(), partial(operator.lt, 3), next_up(3.0), math.inf), (st.floats(), partial(operator.le, 3), 3, math.inf), (st.floats(), partial(operator.eq, 3), 3, 3), (st.floats(), partial(operator.ge, 3), -math.inf, 3), (st.floats(), partial(operator.gt, 3), -math.inf, next_down(3.0)), (st.floats(), lambda x: x < 3, -math.inf, next_down(3.0)), (st.floats(), lambda x: x <= 3, -math.inf, 3), (st.floats(), lambda x: x == 3, 3, 3), (st.floats(), lambda x: x >= 3, 3, math.inf), (st.floats(), lambda x: x > 3, next_up(3.0), math.inf), (st.floats(), lambda x: 3 > x, -math.inf, next_down(3.0)), (st.floats(), lambda x: 3 >= x, -math.inf, 3), (st.floats(), lambda x: 3 == x, 3, 3), (st.floats(), lambda x: 3 <= x, 3, math.inf), (st.floats(), lambda x: 3 < x, next_up(3.0), math.inf), (st.floats(), lambda x: 0 < x < 5, next_up(0.0), next_down(5.0)), (st.floats(), lambda x: 0 < x >= 1, 1, math.inf), (st.floats(), lambda x: 1 > x <= 0, -math.inf, 0), (st.floats(), lambda x: x > 0 and x > 0, next_up(0.0), math.inf), (st.floats(), lambda x: x < 1 and x < 1, -math.inf, next_down(1.0)), (st.floats(), lambda x: x > 1 and x > 0, next_up(1.0), math.inf), (st.floats(), lambda x: x < 1 and x < 2, -math.inf, next_down(1.0))], ids=get_pretty_function_description)
@settings(max_examples=A_FEW)
@given(data=st.data())
def test_filter_rewriting_floats(data, strategy, predicate, min_value, max_value):
    s: LazyStrategy[float] = strategy.filter(predicate)
    assert isinstance(s, LazyStrategy)
    assert isinstance(s.wrapped_strategy, FloatStrategy)
    assert s.wrapped_strategy.min_value == min_value
    assert s.wrapped_strategy.max_value == max_value
    value: float = data.draw(s)
    assert predicate(value)

@pytest.mark.parametrize('pred', [math.isinf, math.isnan, partial(operator.lt, 6), partial(operator.eq, Fraction(10, 3)), partial(operator.ge, 0), partial(operator.lt, math.inf), partial(operator.gt, -math.inf)])
@pytest.mark.parametrize('s', [st.integers(1, 5), st.floats(1, 5)])
def test_rewrite_unsatisfiable_filter(s, pred):
    assert s.filter(pred).is_empty

@pytest.mark.parametrize('pred', [partial(operator.eq, 'numbers are never equal to strings')])
@pytest.mark.parametrize('s', [st.integers(1, 5), st.floats(1, 5)])
@fails_with(Unsatisfiable)
def test_erroring_rewrite_unsatisfiable_filter(s, pred):
    check_can_generate_examples(s.filter(pred))

@pytest.mark.parametrize('strategy, predicate', [(st.floats(), math.isinf), (st.floats(0, math.inf), math.isinf), (st.floats(), math.isnan)])
@given(data=st.data())
def test_misc_sat_filter_rewrites(data