import numpy as np
from pandas import Categorical, DataFrame, Index, Series, NaT
from pandas.api.types import is_scalar
from typing import (
    Any,
    Callable,
    Union,
    List,
    Optional,
    Tuple,
    Dict,
    Sequence,
    Iterable,
    TypeVar,
    overload,
)
from datetime import date, datetime, timedelta
import re
import sys
from pytest import mark

class TestCategoricalAnalytics:
    @mark.parametrize('aggregation', ['min', 'max'])
    def test_min_max_not_ordered_raises(self, aggregation: str) -> None:
        ...

    def test_min_max_ordered(self, index_or_series_or_array: Callable[[Categorical], Union[Index, Series, np.ndarray]]) -> None:
        ...

    def test_min_max_reduce(self) -> None:
        ...

    @mark.parametrize('categories,expected', [(list('ABC'), np.nan), ([1, 2, 3], np.nan), pytest.param(Series(date_range('2020-01-01', periods=3), dtype='category'), NaT, marks=pytest.mark.xfail(reason='https://github.com/pandas-dev/pandas/issues/29962'))])
    @mark.parametrize('aggregation', ['min', 'max'])
    def test_min_max_ordered_empty(self, categories: Union[List[str], List[int], Series], expected: Union[np.nan, NaT], aggregation: str) -> None:
        ...

    @mark.parametrize('values, categories', [(['a', 'b', 'c', np.nan], list('cba')), ([1, 2, 3, np.nan], [3, 2, 1])])
    @mark.parametrize('function', ['min', 'max'])
    def test_min_max_with_nan(self, values: List[Union[str, float]], categories: List[str], function: str, skipna: bool) -> None:
        ...

    @mark.parametrize('function', ['min', 'max'])
    def test_min_max_only_nan(self, function: str, skipna: bool) -> None:
        ...

    @mark.parametrize('method', ['min', 'max'])
    def test_numeric_only_min_max_raises(self, method: str) -> None:
        ...

    @mark.parametrize('method', ['min', 'max'])
    def test_numpy_min_max_raises(self, method: str) -> None:
        ...

    @mark.parametrize('kwarg', ['axis', 'out', 'keepdims'])
    @mark.parametrize('method', ['min', 'max'])
    def test_numpy_min_max_unsupported_kwargs_raises(self, method: str, kwarg: str) -> None:
        ...

    @mark.parametrize('method, expected', [('min', 'a'), ('max', 'c')])
    def test_numpy_min_max_axis_equals_none(self, method: str, expected: str) -> None:
        ...

    @mark.parametrize('values,categories,exp_mode', [([1, 1, 2, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5]), ([1, 1, 1, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5, 1]), ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1]), ([np.nan, np.nan, np.nan, 4, 5], [5, 4, 3, 2, 1], [5, 4]), ([np.nan, np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4]), ([np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4])])
    def test_mode(self, values: List[int], categories: List[int], exp_mode: List[int]) -> None:
        ...

    def test_searchsorted(self, ordered: bool) -> None:
        ...

    def test_unique(self, ordered: bool) -> None:
        ...

    def test_unique_index_series(self, ordered: bool) -> None:
        ...

    def test_shift(self) -> None:
        ...

    def test_nbytes(self) -> None:
        ...

    def test_memory_usage(self, using_infer_string: bool) -> None:
        ...

    def test_map(self) -> None:
        ...

    @mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
    def test_validate_inplace_raises(self, value: Union[int, str, List[int], float]) -> None:
        ...

    def test_quantile_empty(self) -> None:
        ...