import numpy as np
import pytest
from typing import Any

import pandas as pd
from pandas import CategoricalIndex, Series, Timedelta, Timestamp, date_range
from pandas.core.arrays import (
    DatetimeArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)

class TestToIterable:
    dtypes: list[tuple[str, type]]

    @pytest.mark.parametrize('dtype, rdtype', dtypes)
    @pytest.mark.parametrize(
        'method',
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=['tolist', 'to_list', 'list', 'iter'],
    )
    def test_iterable(self, index_or_series: type, method: Any, dtype: str, rdtype: type) -> None: ...

    @pytest.mark.parametrize(
        'dtype, rdtype, obj',
        [
            ('object', object, 'a'),
            ('object', int, 1),
            ('category', object, 'a'),
            ('category', int, 1),
        ],
    )
    @pytest.mark.parametrize(
        'method',
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=['tolist', 'to_list', 'list', 'iter'],
    )
    def test_iterable_object_and_category(self, index_or_series: type, method: Any, dtype: str, rdtype: type, obj: Any) -> None: ...

    @pytest.mark.parametrize('dtype, rdtype', dtypes)
    def test_iterable_items(self, dtype: str, rdtype: type) -> None: ...

    @pytest.mark.parametrize('dtype, rdtype', dtypes + [('object', int), ('category', int)])
    def test_iterable_map(self, index_or_series: type, dtype: str, rdtype: type) -> None: ...

    @pytest.mark.parametrize(
        'method',
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=['tolist', 'to_list', 'list', 'iter'],
    )
    def test_categorial_datetimelike(self, method: Any) -> None: ...

    def test_iter_box_dt64(self, unit: str) -> None: ...
    def test_iter_box_dt64tz(self, unit: str) -> None: ...
    def test_iter_box_timedelta64(self, unit: str) -> None: ...
    def test_iter_box_period(self) -> None: ...

def test_values_consistent(arr: Any, expected_type: type, dtype: Any, using_infer_string: bool) -> None: ...
def test_numpy_array(arr: np.ndarray) -> None: ...
def test_numpy_array_all_dtypes(any_numpy_dtype: Any) -> None: ...
def test_array(arr: Any, attr: str, index_or_series: type) -> None: ...
def test_array_multiindex_raises() -> None: ...
def test_to_numpy(arr: Any, expected: np.ndarray, zero_copy: bool, index_or_series_or_array: type) -> None: ...
def test_to_numpy_copy(arr: np.ndarray, as_series: bool, using_infer_string: bool) -> None: ...
def test_to_numpy_dtype(as_series: bool) -> None: ...
def test_to_numpy_na_value_numpy_dtype(index_or_series: type, values: list[Any], dtype: Any, na_value: Any, expected: list[Any]) -> None: ...
def test_to_numpy_multiindex_series_na_value(data: list[Any], multiindex: list[tuple[Any, ...]], dtype: Any, na_value: Any, expected: list[Any]) -> None: ...
def test_to_numpy_kwargs_raises() -> None: ...
def test_to_numpy_dataframe_na_value(data: dict[str, Any], dtype: type, na_value: Any) -> None: ...
def test_to_numpy_dataframe_single_block(data: dict[str, Any], expected_data: list[list[float]]) -> None: ...
def test_to_numpy_dataframe_single_block_no_mutate() -> None: ...

class TestAsArray:
    def test_asarray_object_dt64(self, tz: str | None) -> None: ...
    def test_asarray_tz_naive(self) -> None: ...
    def test_asarray_tz_aware(self) -> None: ...